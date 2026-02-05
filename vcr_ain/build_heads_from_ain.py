#!/usr/bin/env python3
import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from peft import PeftModel
except Exception:
    PeftModel = None
from qwen_vl_utils import process_vision_info


def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return {}
    return {}


def clean_caption(caption: str) -> str:
    text = caption.strip().rstrip(".!?")
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^personx\b", "", text, flags=re.IGNORECASE).strip()
    if text.lower().startswith(("is ", "are ", "was ", "were ")):
        text = re.sub(r"^(is|are|was|were)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"^(?:\\d+|one|two|three|four|five|six|seven|eight|nine|ten)\\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(man|woman|person|boy|girl|child|guy|lady|men|women|people|persons|children|group|group of|couple|pair)\\b",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if not text:
        return ""
    lower = text.lower()

    preps = (
        "in ",
        "with ",
        "at ",
        "on ",
        "near ",
        "around ",
        "by ",
        "behind ",
        "beside ",
        "under ",
        "over ",
        "inside ",
        "outside ",
        "between ",
        "during ",
        "while ",
    )
    verb_like = (
        "engaged",
        "talking",
        "speaking",
        "standing",
        "sitting",
        "walking",
        "holding",
        "wearing",
        "looking",
        "playing",
        "reading",
        "cooking",
        "eating",
        "drinking",
        "running",
        "smiling",
        "laughing",
        "fighting",
        "kissing",
        "hugging",
        "meeting",
        "celebrating",
        "marrying",
        "signing",
        "dancing",
        "praying",
    )
    first = lower.split()[0]
    if lower.startswith(preps) or (first not in verb_like and not first.endswith("ing")):
        return "PersonX is " + text
    return "PersonX " + text


def build_head(caption: str, question: str) -> str:
    if caption:
        return clean_caption(caption)
    if question:
        q = question.strip().rstrip("?")
        q = re.sub(
            r"^(why|how|what|where|when|who|which|whom)\b",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()
        if not q.lower().startswith("personx"):
            q = "PersonX " + q
        return q
    return ""


def build_messages(image_path: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        "Return JSON with keys: caption, entities.\n"
                        "Use the image above.\n"
                        "caption: one short event sentence starting with 'PersonX is ...' describing a visible action.\n"
                        "entities: 3-8 concrete noun phrases (no sentences).\n"
                        "Do not use placeholders, angle brackets, or template words like 'verbing' or 'noun1'.\n"
                        "Output JSON only."
                    ),
                },
            ],
        },
    ]


def contains_placeholder(text: str) -> bool:
    lower = text.lower()
    if re.search(r"<[^>]+>", text):
        return True
    if "verbing" in lower:
        return True
    if re.search(r"\bnoun\\d+\b", lower):
        return True
    return False


def generate_caption_entities(model, processor, image_path: str) -> Tuple[str, List[str]]:
    messages = build_messages(image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    obj = extract_json(decoded)
    caption = str(obj.get("caption", "")).strip()
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    caption = "" if contains_placeholder(caption) else caption
    cleaned_entities = []
    for e in entities:
        e_str = str(e).strip()
        if not e_str or contains_placeholder(e_str):
            continue
        cleaned_entities.append(e_str)
    return caption, cleaned_entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PersonX heads from AIN captions.")
    parser.add_argument("--input", required=True, help="Input JSONL (from preprocess_vcr).")
    parser.add_argument("--output", required=True, help="Output JSONL with head.")
    parser.add_argument("--model", default="MBZUAI/AIN")
    parser.add_argument("--lora", default="", help="Optional LoRA adapter path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            caption, entities = generate_caption_entities(model, processor, rec.get("img", ""))
            head = build_head(caption, rec.get("question", ""))
            rec["caption"] = caption
            rec["entities"] = entities
            rec["head"] = head
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
