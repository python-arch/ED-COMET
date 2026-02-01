#!/usr/bin/env python3
import argparse
import json
import re


def head_from_question(question: str) -> str:
    q = question.strip()
    q = re.sub(r"\?$", "", q)
    q = re.sub(
        r"^(why|how|what|where|when|who|which|whom)\b", "", q, flags=re.IGNORECASE
    ).strip()
    q = re.sub(
        r"^(is|are|was|were|do|does|did|can|could|would|will|has|have|had|should|might|may)\b",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    if not q:
        return ""
    if not q.lower().startswith("personx"):
        q = "PersonX " + q
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Add naive PersonX heads from questions.")
    parser.add_argument("--input", required=True, help="Input JSONL (QA or QA->R).")
    parser.add_argument("--output", required=True, help="Output JSONL with head field.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            head = head_from_question(rec.get("question", ""))
            rec["head"] = head
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
