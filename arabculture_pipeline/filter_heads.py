#!/usr/bin/env python3
"""Filter low-quality heads before tail generation."""
import argparse
import json
import re
from typing import Dict


ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
PERSONX_ARTICLE_RE = re.compile(r"^PersonX\s+(?:The|A|An)\b", re.IGNORECASE)
ADVERB_RE = re.compile(r"\b(typically|usually|often|generally)\b", re.IGNORECASE)
DEMONYM_RE = re.compile(
    r"^PersonX\s+(?:Egyptian|Jordanian|Palestinian|Saudi|Emirati|UAE|Moroccan|"
    r"Tunisian|Sudanese|Kuwaiti|Qatari|Omani|Bahraini|Lebanese|Syrian|Iraqi|"
    r"Libyan|Algerian|Yemeni)s?\b",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"^PersonX\s+[A-Z][a-z]{2,}\b")


def alpha_ratio(text: str) -> float:
    alnum = [ch for ch in text if ch.isalnum()]
    if not alnum:
        return 0.0
    letters = sum(ch.isalpha() for ch in alnum)
    return letters / max(1, len(alnum))


def is_bad_head(
    head: str,
    reject_non_ascii: bool,
    min_alpha_ratio: float,
    max_words: int,
    reject_other_persons: bool,
    reject_arabic: bool,
    reject_personx_article: bool,
    reject_adverbs: bool,
    reject_demonyms: bool,
) -> bool:
    if "/" in head or "\\" in head:
        return True
    if reject_non_ascii and any(ord(ch) > 127 for ch in head):
        return True
    if reject_arabic and ARABIC_RE.search(head):
        return True
    if reject_other_persons and re.search(r"\bPerson[YZ]\b", head, flags=re.I):
        return True
    if reject_personx_article and PERSONX_ARTICLE_RE.search(head):
        return True
    if reject_adverbs and ADVERB_RE.search(head):
        return True
    if NAME_RE.search(head):
        return True
    if reject_demonyms and DEMONYM_RE.search(head):
        return True
    if max_words and len(head.split()) > max_words:
        return True
    if min_alpha_ratio and alpha_ratio(head) < min_alpha_ratio:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="heads.jsonl")
    parser.add_argument("--output", required=True, help="heads.filtered.jsonl")
    parser.add_argument("--reject-non-ascii", action="store_true")
    parser.add_argument("--reject-arabic", action="store_true")
    parser.add_argument("--reject-other-persons", action="store_true")
    parser.add_argument("--reject-personx-article", action="store_true")
    parser.add_argument("--reject-adverbs", action="store_true")
    parser.add_argument("--reject-demonyms", action="store_true")
    parser.add_argument("--min-alpha-ratio", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=0)
    args = parser.parse_args()

    total = 0
    kept = 0
    dropped = 0

    with open(args.input, "r", encoding="utf-8") as src, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for line in src:
            if not line.strip():
                continue
            total += 1
            obj: Dict[str, str] = json.loads(line)
            head = obj.get("head", "").strip()
            if not head or is_bad_head(
                head,
                args.reject_non_ascii,
                args.min_alpha_ratio,
                args.max_words,
                args.reject_other_persons,
                args.reject_arabic,
                args.reject_personx_article,
                args.reject_adverbs,
                args.reject_demonyms,
            ):
                dropped += 1
                continue
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"total={total} kept={kept} dropped={dropped}")


if __name__ == "__main__":
    main()
