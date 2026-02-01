#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def generate_tails(
    model,
    tokenizer,
    prompt: str,
    num: int,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        num_return_sequences=num,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.strip() for d in decoded if d.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ED-COMET cache from heads.")
    parser.add_argument("--input", required=True, help="JSONL with `head` and optional `tags`.")
    parser.add_argument("--output", required=True, help="Output cache JSONL.")
    parser.add_argument("--model", required=True, help="HF model path for ED-COMET.")
    parser.add_argument("--tokenizer", default="", help="Tokenizer path (optional).")
    parser.add_argument("--tags", default="<REGION=MENA> <COUNTRY=EGY>")
    parser.add_argument(
        "--relations",
        default="xIntent,xNeed,xReact,xEffect,oReact,oEffect",
        help="Comma-separated relations.",
    )
    parser.add_argument("--num-tails", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    relations = [r.strip() for r in args.relations.split(",") if r.strip()]

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            head = rec.get("head", "").strip()
            if not head:
                continue
            tags = rec.get("tags", args.tags)
            out: Dict[str, List[str]] = {"event": head, "meta": {"tags": tags}}
            for rel in relations:
                prompt = f"{tags} {head} {rel}"
                tails = generate_tails(
                    model,
                    tokenizer,
                    prompt,
                    num=args.num_tails,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                out[rel] = tails
            fout.write(json.dumps(out, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
