#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict


def to_qwen_messages(prompt: str, image_path: str, target: str, system: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": target},
        ]
    }


def to_simple(prompt: str, image_path: str, target: str) -> Dict[str, Any]:
    return {"image": image_path, "prompt": prompt, "target": target}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AIN SFT JSONL from VCR JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL (vcr_qa or vcr_qar).")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--format", choices=["qwen", "simple"], default="qwen")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System message for Qwen-style messages.",
    )
    args = parser.parse_args()

    out_count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            prompt = rec.get("augmented_prompt") or rec.get("prompt", "")
            image = rec.get("img")
            target = rec.get("target", "")

            if args.format == "qwen":
                out = to_qwen_messages(prompt, image, target, args.system)
            else:
                out = to_simple(prompt, image, target)

            out["qid"] = rec.get("qid")
            out["task"] = rec.get("task")
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")
            out_count += 1

    print(f"Wrote {out_count} records to {args.output}")


if __name__ == "__main__":
    main()
