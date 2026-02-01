#!/usr/bin/env python3
"""Repair low-quality heads in atomic_full or heads jsonl."""
import argparse
import json
import logging
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover
    raise SystemExit("vLLM is required. Install it first.") from exc

LOGGER = logging.getLogger("repair_heads")

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
PERSONX_ARTICLE_RE = re.compile(r"^PersonX\s+(?:The|A|An)\b", re.IGNORECASE)
ADVERB_RE = re.compile(r"\b(typically|usually|often|generally)\b", re.IGNORECASE)
DEMONYM_RE = re.compile(
    r"^PersonX\s+(?:Egyptian|Jordanian|Palestinian|Saudi|Emirati|UAE|Moroccan|"
    r"Tunisian|Sudanese|Kuwaiti|Qatari|Omani|Bahraini|Lebanese|Syrian|Iraqi|"
    r"Libyan|Algerian|Yemeni)\b",
    re.IGNORECASE,
)


def extract_json(text: str):
    candidates = []
    depth = 0
    start = None
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
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
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def normalize_head(head: str) -> str:
    head = head.strip().rstrip(".!?")
    head = re.sub(r"^personx[\s:,-]*", "", head, flags=re.I).strip()
    if head:
        head = re.split(r"[.!?]", head, maxsplit=1)[0].strip()
        head = "PersonX " + head
    return head


def looks_suspicious(head: str, max_words: int) -> bool:
    if "/" in head or "\\" in head:
        return True
    if ARABIC_RE.search(head):
        return True
    if PERSONX_ARTICLE_RE.search(head):
        return True
    if DEMONYM_RE.search(head):
        return True
    if ADVERB_RE.search(head):
        return True
    if max_words and len(head.split()) > max_words:
        return True
    if not head.lower().startswith("personx"):
        return True
    return False


def format_chat(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-head-words", type=int, default=16)
    parser.add_argument("--drop-on-fail", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    rows: List[Dict] = []
    targets: List[Tuple[int, str]] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    for idx, row in enumerate(rows):
        head = row.get("event") or row.get("head") or ""
        head = str(head).strip()
        if looks_suspicious(head, args.max_head_words):
            targets.append((idx, head))

    LOGGER.info("rows=%d suspicious_heads=%d", len(rows), len(targets))
    if not targets:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )

    system = (
        "You are a data correction engine. Return ONLY valid JSON. "
        "Rewrite the head into a single English event. Start with PersonX + verb. "
        "No Arabic. No 'PersonX The/A/An'. No statistics or 'typically/usually'."
    )

    prompts = []
    for _, head in targets:
        user = (
            "Rewrite into clean ATOMIC head. Return JSON only:\n"
            "{\"head\":\"PersonX ...\"}\n"
            f"HEAD: {head}"
        )
        prompts.append(format_chat(tokenizer, system, user))

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
    repaired = 0
    dropped = 0
    total = len(prompts)

    for start in range(0, total, args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, params)
        for out, (idx, old_head) in zip(outputs, targets[start : start + args.batch_size]):
            obj = extract_json(out.outputs[0].text)
            if not obj or "head" not in obj:
                if args.drop_on_fail:
                    rows[idx]["_drop"] = True
                    dropped += 1
                continue
            new_head = normalize_head(str(obj["head"]))
            if looks_suspicious(new_head, args.max_head_words) or re.search(
                r"\bPerson[YZ]\b", new_head, flags=re.I
            ):
                if args.drop_on_fail:
                    rows[idx]["_drop"] = True
                    dropped += 1
                continue
            if "event" in rows[idx]:
                rows[idx]["event"] = new_head
            else:
                rows[idx]["head"] = new_head
            repaired += 1
        LOGGER.info("Processed %d/%d", min(start + args.batch_size, total), total)

    LOGGER.info("repaired=%d dropped=%d", repaired, dropped)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            if row.get("_drop"):
                continue
            row.pop("_drop", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
