#!/usr/bin/env python3
"""Repair suspicious tails using a lightweight LLM pass."""
import argparse
import json
import logging
import os
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover
    raise SystemExit("vLLM is required. Install it first.") from exc

RELATIONS = [
    "xIntent",
    "xNeed",
    "xEffect",
    "xReact",
    "xWant",
    "xAttr",
    "oEffect",
    "oReact",
    "oWant",
]

SHORT_OK = {"to", "a", "an", "i", "of", "in", "on", "at", "is", "be", "as", "by"}
TO_RELATIONS = {"xIntent", "xNeed", "xWant", "oWant"}
EFFECT_RELATIONS = {"xEffect", "oEffect"}
REACT_RELATIONS = {"xReact", "oReact"}

LOGGER = logging.getLogger("repair_tails")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def alpha_ratio(text: str) -> float:
    alnum = [ch for ch in text if ch.isalnum()]
    if not alnum:
        return 0.0
    letters = sum(ch.isalpha() for ch in alnum)
    return letters / max(1, len(alnum))


def looks_suspicious(text: str) -> bool:
    if "/" in text or "\\" in text:
        return True
    if "personx" in text.lower() or "persony" in text.lower() or "personz" in text.lower():
        return True
    if alpha_ratio(text) < 0.6:
        return True
    tokens = normalize_text(text).split()
    short_bad = [tok for tok in tokens if len(tok) <= 2 and tok not in SHORT_OK]
    if short_bad and len(tokens) <= 3:
        return True
    if len(short_bad) >= 2:
        return True
    return False


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


def validate_tail(rel: str, tail: str) -> bool:
    text = tail.strip()
    if not text:
        return False
    lower = text.lower()
    if rel in TO_RELATIONS and not lower.startswith("to "):
        return False
    if rel in EFFECT_RELATIONS | REACT_RELATIONS and lower.startswith("to "):
        return False
    if rel == "xAttr" and len(text.split()) > 3:
        return False
    return True


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
    parser.add_argument("--input", required=True, help="atomic_full.*.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    rows: List[Dict] = []
    repair_map: Dict[Tuple[int, str, int], str] = {}
    original_map: Dict[Tuple[int, str, int], str] = {}
    total = 0
    flagged = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)

    for row_idx, row in enumerate(rows):
        for rel in RELATIONS:
            tails = row.get(rel, []) or []
            for t_idx, tail in enumerate(tails):
                total += 1
                if looks_suspicious(tail):
                    flagged += 1
                    repair_map[(row_idx, rel, t_idx)] = tail
                    original_map[(row_idx, rel, t_idx)] = tail

    LOGGER.info("total tails=%d flagged=%d", total, flagged)
    if args.dry_run or not repair_map:
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
        "You are a text correction engine. Return ONLY valid JSON. "
        "Fix spelling, casing, and minor grammar only. Do not change meaning. "
        "Remove PersonX/PersonY/PersonZ tokens if present and keep the word 'to' intact."
    )

    prompts = []
    keys = list(repair_map.keys())
    for key in keys:
        tail = repair_map[key]
        user = (
            "Fix spelling/grammar only. Return JSON: {\"tail\": \"...\"}.\n"
            f"TAIL: {tail}"
        )
        prompts.append(format_chat(tokenizer, system, user))

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    repaired = 0
    invalid_repairs = 0
    total_prompts = len(prompts)
    for start in range(0, total_prompts, args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, params)
        for out, key in zip(outputs, keys[start : start + args.batch_size]):
            obj = extract_json(out.outputs[0].text)
            if obj and "tail" in obj:
                new_tail = obj["tail"].strip()
                if looks_suspicious(new_tail) or not validate_tail(key[1], new_tail):
                    repair_map[key] = original_map[key]
                    invalid_repairs += 1
                else:
                    repair_map[key] = new_tail
                    repaired += 1
        LOGGER.info("Repaired %d/%d", min(start + args.batch_size, total_prompts), total_prompts)

    LOGGER.info("repairs applied=%d invalid_repairs=%d", repaired, invalid_repairs)

    for (row_idx, rel, t_idx), new_tail in repair_map.items():
        rows[row_idx][rel][t_idx] = new_tail

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
