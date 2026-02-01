#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List


def obj_name(obj: str, idx: int) -> str:
    name = re.sub(r"\s+", "_", obj.strip())
    return f"{name}_{idx}"


def render_tokens(tokens: Iterable[Any], objects: List[str]) -> str:
    parts: List[str] = []
    for tok in tokens:
        if isinstance(tok, list):
            names = [obj_name(objects[i], i) for i in tok]
            if len(names) == 1:
                parts.append(names[0])
            elif len(names) == 2:
                parts.append(f"{names[0]} and {names[1]}")
            else:
                parts.append(", ".join(names[:-1]) + ", and " + names[-1])
        else:
            parts.append(str(tok))
    text = " ".join(parts)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+'s", "'s", text)
    text = re.sub(r"\s+n't", "n't", text)
    text = re.sub(r"\s+\"([^\"]+)\s+\"", r' "\1"', text)
    return text.strip()


def build_qa_record(item: Dict[str, Any], images_dir: str, idx: int) -> Dict[str, Any]:
    objects = item.get("objects", [])
    q_text = render_tokens(item["question"], objects)
    options = [render_tokens(t, objects) for t in item["answer_choices"]]
    letters = ["A", "B", "C", "D"]
    prompt = "Question: " + q_text + "\nOptions: " + " ".join(
        [f"{letters[i]}) {options[i]}" for i in range(len(options))]
    )
    label = letters[item["answer_label"]]
    qid = item.get("annot_id", item.get("question_id", idx))
    return {
        "qid": qid,
        "img": os.path.join(images_dir, item["img_fn"]),
        "question": q_text,
        "options": {letters[i]: options[i] for i in range(len(options))},
        "prompt": prompt,
        "target": label,
        "task": "qa",
    }


def build_qar_record(item: Dict[str, Any], images_dir: str, idx: int) -> Dict[str, Any]:
    objects = item.get("objects", [])
    q_text = render_tokens(item["question"], objects)
    answers = [render_tokens(t, objects) for t in item["answer_choices"]]
    answer_label = item["answer_label"]
    chosen_answer = answers[answer_label]
    rationales = [render_tokens(t, objects) for t in item["rationale_choices"]]
    letters = ["A", "B", "C", "D"]
    prompt = (
        "Question: " + q_text + "\nChosen Answer: " + chosen_answer + "\nRationale Options: "
        + " ".join([f"{letters[i]}) {rationales[i]}" for i in range(len(rationales))])
    )
    label = letters[item["rationale_label"]]
    qid = item.get("annot_id", item.get("question_id", idx))
    return {
        "qid": qid,
        "img": os.path.join(images_dir, item["img_fn"]),
        "question": q_text,
        "chosen_answer": chosen_answer,
        "options": {letters[i]: rationales[i] for i in range(len(rationales))},
        "prompt": prompt,
        "target": label,
        "task": "qar",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess VCR JSONL into AIN-friendly JSONL.")
    parser.add_argument("--vcr-jsonl", required=True, help="Path to VCR jsonl (train/val/test).")
    parser.add_argument("--images-dir", required=True, help="Path to vcr1images directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSONL files.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on examples.")
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Also write QA->R dataset (rationale choices).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    qa_path = os.path.join(args.out_dir, "vcr_qa.jsonl")
    qar_path = os.path.join(args.out_dir, "vcr_qar.jsonl")

    qa_count = 0
    qar_count = 0
    with open(args.vcr_jsonl, "r", encoding="utf-8") as f, open(
        qa_path, "w", encoding="utf-8"
    ) as qa_out, open(qar_path, "w", encoding="utf-8") as qar_out:
        for idx, line in enumerate(f):
            if args.max_examples and idx >= args.max_examples:
                break
            item = json.loads(line)
            qa_rec = build_qa_record(item, args.images_dir, idx)
            qa_out.write(json.dumps(qa_rec, ensure_ascii=True) + "\n")
            qa_count += 1

            if args.include_rationale:
                qar_rec = build_qar_record(item, args.images_dir, idx)
                qar_out.write(json.dumps(qar_rec, ensure_ascii=True) + "\n")
                qar_count += 1

    print(f"Wrote {qa_count} QA examples to {qa_path}")
    if args.include_rationale:
        print(f"Wrote {qar_count} QA->R examples to {qar_path}")


if __name__ == "__main__":
    main()
