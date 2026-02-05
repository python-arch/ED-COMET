#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from peft import PeftModel
except Exception:
    PeftModel = None
from qwen_vl_utils import process_vision_info


def build_messages(prompt: str, image_path: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def generate_letter(model, processor, prompt: str, image_path: str, max_new_tokens: int) -> str:
    messages = build_messages(prompt, image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    letters = re.findall(r"\b([A-D])\b", decoded)
    if letters:
        return letters[-1]
    return decoded.strip().split()[-1] if decoded.strip() else ""


def score_letters(
    model, processor, prompt: str, image_path: str, letters: List[str]
) -> Tuple[str, Dict[str, float]]:
    messages = build_messages(prompt, image_path)
    base_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[base_text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)

    scores: Dict[str, float] = {}
    for letter in letters:
        target = letter
        with torch.no_grad():
            target_ids = processor.tokenizer(
                target, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(model.device)
            labels = torch.full(
                (1, inputs.input_ids.size(1) + target_ids.size(1)), -100, device=model.device
            )
            concat_input_ids = torch.cat([inputs.input_ids, target_ids], dim=1)
            labels[0, inputs.input_ids.size(1) :] = target_ids[0]
            outputs = model(input_ids=concat_input_ids, labels=labels)
            scores[letter] = -float(outputs.loss)
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIN inference on VCR JSONL.")
    parser.add_argument("--input", required=True, help="JSONL with prompt/augmented_prompt.")
    parser.add_argument("--output", required=True, help="Output JSONL with predictions.")
    parser.add_argument("--model", default="MBZUAI/AIN")
    parser.add_argument("--lora", default="", help="Optional LoRA adapter path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--mode", choices=["generate", "score"], default="generate")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    if args.lora:
        if PeftModel is None:
            raise RuntimeError("peft is not available but --lora was provided.")
        model = PeftModel.from_pretrained(model, args.lora)
        model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    letters = ["A", "B", "C", "D"]
    total = 0
    correct = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            prompt = rec.get("augmented_prompt") or rec.get("prompt", "")
            image_path = rec.get("img")
            target = rec.get("target")

            if args.mode == "score":
                pred, scores = score_letters(model, processor, prompt, image_path, letters)
                rec["scores"] = scores
            else:
                pred = generate_letter(model, processor, prompt, image_path, args.max_new_tokens)

            rec["pred"] = pred
            if target is not None:
                rec["correct"] = int(pred == target)
                correct += rec["correct"]
            total += 1
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if total and target is not None:
        acc = correct / total
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
