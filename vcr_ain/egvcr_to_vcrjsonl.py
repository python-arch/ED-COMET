#!/usr/bin/env python3
import argparse
import ast
import json
from typing import Any, List

from datasets import load_dataset


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EG-VCR split to VCR-style JSONL (Q->A).")
    parser.add_argument("--dataset", default="CulTex-VLM/EG-VCR")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--id-field",
        default="index",
        help="Field to use as qid (use 'index' for dataset index).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)

    letters = ["A", "B", "C", "D"]
    total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(dataset):
            if args.max_examples and idx >= args.max_examples:
                break
            q_text = ex.get("question_orig") or ex.get("question") or ""
            choices = parse_answer_choices(ex.get("answer_choices"))
            if len(choices) != 4:
                continue
            prompt = "Question: " + q_text + "\nOptions: " + " ".join(
                [f"{letters[i]}) {choices[i]}" for i in range(len(choices))]
            )
            answer_label = ex.get("answer_label")
            target = letters[answer_label] if isinstance(answer_label, int) else ""
            if args.id_field == "index":
                qid = idx
            else:
                qid = ex.get(args.id_field, idx)
            rec = {
                "index": idx,
                "qid": qid,
                "question": q_text,
                "options": {letters[i]: choices[i] for i in range(len(choices))},
                "prompt": prompt,
                "target": target,
                "task": "qa",
            }
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            total += 1

    print(f"Wrote {total} QA records to {args.output}")


if __name__ == "__main__":
    main()
