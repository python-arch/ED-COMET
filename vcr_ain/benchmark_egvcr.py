#!/usr/bin/env python3
import argparse
import ast
import json
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def safe_literal_eval(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return default
    return value


def parse_answer_choices(answer_choices: Any) -> List[str]:
    choices = safe_literal_eval(answer_choices, [])
    if not isinstance(choices, list):
        return []
    parsed: List[str] = []
    for choice in choices:
        if isinstance(choice, list):
            parsed.append(" ".join(str(x) for x in choice))
        else:
            parsed.append(str(choice))
    return parsed


def preprocess_image(entry: Dict[str, Any], draw_boxes: bool) -> Image.Image:
    img = entry.get("img_fn") or entry.get("image") or entry.get("img")
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    if not isinstance(img, Image.Image):
        raise ValueError("Could not load image from entry.")
    if not draw_boxes:
        return img

    question = safe_literal_eval(entry.get("question"), [])
    boxes = safe_literal_eval(entry.get("boxes"), [])
    if not isinstance(question, list) or not isinstance(boxes, list):
        return img

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for word in question:
        if isinstance(word, list) and word:
            idx = word[0]
            if isinstance(idx, int) and 0 <= idx < len(boxes):
                box = boxes[idx]
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                label = f"person{idx + 1}"
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1, y1 - 10), label, fill="white")
    return img_copy


def build_prompt(question: str, choices: List[str], label_format: str) -> str:
    if label_format == "digits":
        choices_text = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        target_label = "0, 1, 2, or 3"
    else:
        letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{letters[i]}: {c}" for i, c in enumerate(choices)])
        target_label = "A, B, C, or D"

    return (
        f"{question}\n{choices_text}\n\n"
        "INSTRUCTIONS:\n"
        "- Choose the correct answer.\n"
        f"- Reply with ONLY ONE {('DIGIT' if label_format == 'digits' else 'LETTER')}: {target_label}.\n"
        "- Do NOT include words, explanations, punctuation, or markdown.\n"
        "- The response must be exactly one character long."
    )


def extract_prediction(text: str, label_format: str) -> str:
    if "assistant" in text:
        text = text.split("assistant", 1)[-1]
    if label_format == "digits":
        matches = re.findall(r"\b([0-3])\b", text)
        if matches:
            return matches[-1]
    else:
        matches = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    stripped = text.strip()
    return stripped[-1] if stripped else ""


def generate_choice(
    model,
    processor,
    prompt: str,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.decode(outputs[0], skip_special_tokens=True)
    return decoded


def load_edcomet_prompts(
    path: str,
    id_field: str,
    prompt_field: str,
) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if id_field == "index":
                key = rec.get("index", rec.get("qid"))
            else:
                key = rec.get(id_field)
            if key is None:
                continue
            prompt = rec.get(prompt_field)
            if prompt:
                prompts[str(key)] = prompt
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AIN on EG-VCR (Q->A).")
    parser.add_argument("--dataset", default="CulTex-VLM/EG-VCR")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="MBZUAI/AIN")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--label-format", choices=["digits", "letters"], default="letters")
    parser.add_argument("--mode", choices=["baseline", "edcomet", "both"], default="baseline")
    parser.add_argument("--output", default="")
    parser.add_argument("--id-field", default="index")
    parser.add_argument("--edcomet-jsonl", default="")
    parser.add_argument("--edcomet-prompt-field", default="augmented_prompt")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    edcomet_prompts: Dict[str, str] = {}
    if args.edcomet_jsonl:
        edcomet_prompts = load_edcomet_prompts(
            args.edcomet_jsonl, args.id_field, args.edcomet_prompt_field
        )

    dataset = load_dataset(args.dataset, split=args.split)
    letters = ["A", "B", "C", "D"]

    def target_label(answer_label: Any) -> str:
        if not isinstance(answer_label, int):
            return ""
        return str(answer_label) if args.label_format == "digits" else letters[answer_label]

    base_records: List[Dict[str, Any]] = []
    ed_records: List[Dict[str, Any]] = []
    base_correct = 0
    base_total = 0
    ed_correct = 0
    ed_total = 0
    ed_missing = 0

    for idx, ex in enumerate(dataset):
        if args.max_examples and idx >= args.max_examples:
            break
        question = ex.get("question_orig") or ex.get("question") or ""
        choices = parse_answer_choices(ex.get("answer_choices"))
        if len(choices) != 4:
            continue
        prompt = build_prompt(question, choices, args.label_format)
        image = preprocess_image(ex, args.draw_boxes)
        qid = idx if args.id_field == "index" else ex.get(args.id_field, idx)
        gold = target_label(ex.get("answer_label"))

        if args.mode in ("baseline", "both"):
            decoded = generate_choice(model, processor, prompt, image, args.max_new_tokens)
            pred = extract_prediction(decoded, args.label_format)
            correct = int(pred == gold)
            base_correct += correct
            base_total += 1
            if args.output:
                base_records.append(
                    {
                        "qid": qid,
                        "pred": pred,
                        "gold": gold,
                        "correct": correct,
                        "prompt": prompt,
                    }
                )

        if args.mode in ("edcomet", "both"):
            key = str(qid)
            ed_prompt = edcomet_prompts.get(key)
            if not ed_prompt:
                ed_missing += 1
                continue
            decoded = generate_choice(model, processor, ed_prompt, image, args.max_new_tokens)
            pred = extract_prediction(decoded, args.label_format)
            correct = int(pred == gold)
            ed_correct += correct
            ed_total += 1
            if args.output:
                ed_records.append(
                    {
                        "qid": qid,
                        "pred": pred,
                        "gold": gold,
                        "correct": correct,
                        "prompt": ed_prompt,
                    }
                )

    if args.mode in ("baseline", "both") and base_total:
        acc = base_correct / base_total
        print(f"Baseline accuracy: {acc:.4f} ({base_correct}/{base_total})")
    if args.mode in ("edcomet", "both") and ed_total:
        acc = ed_correct / ed_total
        print(f"ED-COMET accuracy: {acc:.4f} ({ed_correct}/{ed_total})")
    if args.mode in ("edcomet", "both") and ed_missing:
        print(f"ED-COMET prompts missing: {ed_missing}")

    if args.output:
        if args.mode == "baseline":
            out_path = args.output
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in base_records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        elif args.mode == "edcomet":
            out_path = args.output
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in ed_records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        else:
            base_path = args.output + ".baseline.jsonl"
            ed_path = args.output + ".edcomet.jsonl"
            with open(base_path, "w", encoding="utf-8") as f:
                for rec in base_records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            with open(ed_path, "w", encoding="utf-8") as f:
                for rec in ed_records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
