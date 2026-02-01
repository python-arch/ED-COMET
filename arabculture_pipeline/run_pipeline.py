#!/usr/bin/env python3
"""Generate ATOMIC-style data from ArabCulture with Jais (heads) + Qwen (tails)."""
import argparse
import gc
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover - runtime import for user env
    raise SystemExit(
        "vLLM is required. Install it (e.g., `pip install vllm`) before running."
    ) from exc

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
TO_RELATIONS = {"xIntent", "xNeed", "xWant", "oWant"}
EFFECT_RELATIONS = {"xEffect", "oEffect"}
REACT_RELATIONS = {"xReact", "oReact"}
ATTR_RELATIONS = {"xAttr"}

GENERIC_PHRASES = {
    "feel happy",
    "feel sad",
    "feel upset",
    "feel angry",
    "feel good",
    "feel bad",
    "be happy",
    "be sad",
    "be upset",
    "be angry",
    "be okay",
    "be fine",
    "feel okay",
    "feel fine",
    "none",
}
GENERIC_EMOTION_RE = re.compile(
    r"\b(feel|feels|felt)\s+(happy|sad|good|bad|upset|angry)\b",
    re.IGNORECASE,
)

COUNTRY_CODE = {
    "Egypt": "EGY",
    "Jordan": "JOR",
    "Palestine": "PSE",
    "KSA": "KSA",
    "Saudi Arabia": "KSA",
    "UAE": "UAE",
    "United Arab Emirates": "UAE",
    "Morocco": "MAR",
    "Tunisia": "TUN",
    "Sudan": "SDN",
}

LOGGER = logging.getLogger("arabculture_pipeline")


@dataclass
class Example:
    sample_id: str
    country: str
    region: str
    sub_topic: str
    scenario: str


@dataclass
class HeadResult:
    example: Example
    head: str
    tags: str


@dataclass
class TailResult:
    head_result: HeadResult
    tails: Dict[str, List[str]]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


GENERIC_NORM = {normalize_text(x) for x in GENERIC_PHRASES}


def strip_bullets(text: str) -> str:
    return re.sub(r"^[\\s\\-\\*\\d\\)\\.\\u2022•]+", "", text).strip()


def clean_tail(text: str) -> str:
    text = strip_bullets(text)
    text = text.strip().strip(" .;:!?,")
    return text


def tail_too_short(text: str, rel: str) -> bool:
    norm_tokens = normalize_text(text).split()
    if not norm_tokens:
        return True
    if len(norm_tokens) == 1 and len(norm_tokens[0]) < 3:
        return True
    if rel in {"xIntent", "xNeed", "xWant", "oWant"} and len(norm_tokens) < 2:
        return True
    return len(text.strip()) < 3


def tail_low_alpha_ratio(text: str, threshold: float) -> bool:
    alnum = [ch for ch in text if ch.isalnum()]
    if not alnum:
        return True
    letters = sum(ch.isalpha() for ch in alnum)
    return (letters / max(1, len(alnum))) < threshold


@dataclass
class QualityRules:
    require_to: bool
    react_min_tokens: int
    alpha_ratio: float


def get_quality_rules(mode: str) -> QualityRules:
    if mode == "strict":
        return QualityRules(require_to=True, react_min_tokens=3, alpha_ratio=0.7)
    return QualityRules(require_to=False, react_min_tokens=2, alpha_ratio=0.6)


def normalize_tail_for_relation(rel: str, text: str, rules: QualityRules) -> str:
    text = clean_tail(text)
    if not text:
        return ""
    norm = normalize_text(text)
    if rel in TO_RELATIONS and not norm.startswith("to "):
        text = "to " + text
    return text


def jaccard_overlap(a: str, b: str) -> float:
    def tokens(s: str) -> set:
        return {t for t in normalize_text(s).split() if t not in {"personx", "persony", "personz"}}

    a_set = tokens(a)
    b_set = tokens(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def extract_json(text: str) -> Optional[dict]:
    # Brace-balanced extraction that skips braces inside quoted strings.
    candidates = []
    start = None
    depth = 0
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        else:
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : idx + 1])
                        start = None
    # Fallback to non-greedy regex if no balanced blocks.
    if not candidates:
        candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


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


def load_arabculture(
    countries: Iterable[str],
    keep_country_yes: bool,
    keep_discard_no: bool,
):
    examples: List[Example] = []
    stats = {
        "total": 0,
        "kept": 0,
        "filtered_discard": 0,
        "filtered_country": 0,
    }
    for country in countries:
        ds = load_dataset("MBZUAI/ArabCulture", country, split="test")
        LOGGER.info("Loaded %s rows for %s", len(ds), country)
        for row in ds:
            stats["total"] += 1
            if keep_discard_no and str(row.get("should_discard", "No")).strip() != "No":
                stats["filtered_discard"] += 1
                continue
            if keep_country_yes and str(row.get("relevant_to_this_country", "Yes")).strip() != "Yes":
                stats["filtered_country"] += 1
                continue
            options = row.get("options", {}) or {}
            answer_key = row.get("answer_key", {}) or {}
            arabic_key = str(answer_key.get("arabic_answer_key", "")).strip()
            english_key = str(answer_key.get("english_answer_key", "")).strip()
            arabic_keys = options.get("arabic_keys", []) or []
            english_keys = options.get("english_keys", []) or []
            option_texts = options.get("text", []) or []
            correct_opt = ""
            if arabic_key and arabic_keys and arabic_key in arabic_keys:
                idx = arabic_keys.index(arabic_key)
                if idx < len(option_texts):
                    correct_opt = option_texts[idx]
            elif english_key and english_keys and english_key in english_keys:
                idx = english_keys.index(english_key)
                if idx < len(option_texts):
                    correct_opt = option_texts[idx]
            scenario = row.get("first_statement", "").strip()
            if correct_opt:
                scenario = f"{scenario}. {correct_opt.strip()}"
            examples.append(
                Example(
                    sample_id=str(row.get("sample_id", "")),
                    country=str(row.get("country", country)),
                    region=str(row.get("region", "")),
                    sub_topic=str(row.get("sub_topic", "")),
                    scenario=scenario,
                )
            )
            stats["kept"] += 1
    return examples, stats


def build_tags(country: str, region: str, add_region: bool, add_country: bool) -> str:
    tags = []
    if add_region:
        tags.append("<REGION=MENA>")
    if add_country:
        code = COUNTRY_CODE.get(country, country[:3].upper())
        tags.append(f"<COUNTRY={code}>")
    return " ".join(tags)


def make_llm(model_id: str, tp_size: int, max_model_len: int, gpu_mem_util: float):
    return LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
    )


def generate_heads(
    examples: List[Example],
    model_id: str,
    tp_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    add_region: bool,
    add_country: bool,
    temperature: float,
    batch_size: int,
    max_head_words: int,
) -> List[HeadResult]:
    system = (
        "You are a strict data generation engine. Return ONLY valid JSON. No extra text."
    )
    user_tpl = (
        "Convert this Arabic scenario into ONE ATOMIC-style head event in English.\n\n"
        "Country: {country}\n"
        "Scenario (MSA): {scenario}\n\n"
        "Return ONLY JSON:\n"
        "{{\"head\":\"PersonX ...\"}}\n\n"
        "Rules:\n"
        "- One event, one sentence.\n"
        "- No stereotypes, no moral judgement.\n"
        "- Include cultural details only if explicitly implied.\n"
        "- Start with \"PersonX\" followed by a verb (not \"PersonX The/A/An\").\n"
        "- Do not end the head with punctuation.\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = make_llm(model_id, tp_size, max_model_len, gpu_mem_util)
    prompts = []
    for ex in examples:
        user = user_tpl.format(country=ex.country, scenario=ex.scenario)
        prompts.append(format_chat(tokenizer, system, user))

    params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=128)
    outputs = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        outputs.extend(llm.generate(batch, params))
        LOGGER.info("Heads: %d/%d", min(start + batch_size, total), total)

    results: List[HeadResult] = []
    parse_fail = 0
    length_drop = 0
    for ex, out in zip(examples, outputs):
        text = out.outputs[0].text
        obj = extract_json(text)
        if not obj or "head" not in obj:
            parse_fail += 1
            continue
        head = obj["head"].strip()
        if head:
            head = head.strip().rstrip(".!?")
            head = re.sub(r"^personx[\s:,-]*", "", head, flags=re.I).strip()
            if head:
                head = re.split(r"[.!?]", head, maxsplit=1)[0].strip()
                head = "PersonX " + head
        if head and max_head_words:
            if len(head.split()) > max_head_words:
                length_drop += 1
                continue
        if not head:
            continue
        tags = build_tags(ex.country, ex.region, add_region, add_country)
        results.append(HeadResult(example=ex, head=head, tags=tags))

    LOGGER.info(
        "Heads parsed: %d/%d (failed %d, length_drop %d)",
        len(results),
        len(outputs),
        parse_fail,
        length_drop,
    )

    del llm
    gc.collect()
    return results


def build_tails_prompt(head: str, tags: str, n_tails: int) -> str:
    return (
        "Generate ATOMIC inferences for the head.\n\n"
        f"HEAD: {head}\n"
        f"TAGS: {tags}\n\n"
        "Return ONLY JSON with keys:\n"
        "xIntent,xNeed,xEffect,xReact,xWant,xAttr,oEffect,oReact,oWant\n\n"
        f"Constraints:\n- Exactly {n_tails} tails per key.\n"
        "- Each key must map to a JSON array of plain strings (no bullets or numbering).\n"
        "- Each tail must be at least 2 words (xAttr can be 1-3 words).\n"
        "- Short (3-10 words), verb phrases preferred.\n"
        "- Distinct tails; no paraphrases of the head.\n"
        "- Avoid generic emotion phrases like \"feels happy/sad/good/bad\" unless grounded by a specific reason.\n"
        "- No \"none\".\n"
        "- Each key must map to a JSON array of strings.\n"
        "- xIntent/xNeed/xWant/oWant should start with \"to\".\n"
        "- xEffect/oEffect should be outcomes (no \"to ...\" advice).\n"
        "- xReact/oReact should be feelings or thoughts.\n"
        "- xAttr should be short traits (1-3 words, adjectives).\n"
        "- Avoid single-letter outputs or list markers.\n"
        "- Before returning, verify: valid JSON, all 9 keys present, each key has exactly N items, no single-letter items.\n"
    )


def filter_tails(
    head: str,
    tails: Dict[str, List[str]],
    min_tails: int,
    overlap: float,
    rules: QualityRules,
) -> Optional[Dict[str, List[str]]]:
    cleaned: Dict[str, List[str]] = {}
    for rel in RELATIONS:
        rel_tails = []
        seen_norm = set()
        for t in tails.get(rel, []):
            t = normalize_tail_for_relation(rel, t, rules)
            if not t:
                continue
            if tail_too_short(t, rel):
                continue
            if GENERIC_EMOTION_RE.search(t):
                continue
            if tail_low_alpha_ratio(t, rules.alpha_ratio):
                continue
            if normalize_text(t) in GENERIC_NORM:
                continue
            norm = normalize_text(t)
            if rel in EFFECT_RELATIONS | REACT_RELATIONS and norm.startswith("to "):
                continue
            if rel in ATTR_RELATIONS:
                if len(norm.split()) > 3:
                    continue
            if rel in REACT_RELATIONS and len(norm.split()) < rules.react_min_tokens:
                continue
            if jaccard_overlap(head, t) >= overlap:
                continue
            if t.lower() == "none":
                continue
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            if t not in rel_tails:
                rel_tails.append(t)
        if len(rel_tails) < min_tails:
            return None
        cleaned[rel] = rel_tails[:min_tails]
    return cleaned


def filter_tails_partial(
    head: str,
    tails: Dict[str, List[str]],
    overlap: float,
    rules: QualityRules,
) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for rel in RELATIONS:
        rel_tails = []
        seen_norm = set()
        for t in tails.get(rel, []) or []:
            t = normalize_tail_for_relation(rel, t, rules)
            if not t:
                continue
            if tail_too_short(t, rel):
                continue
            if GENERIC_EMOTION_RE.search(t):
                continue
            if tail_low_alpha_ratio(t, rules.alpha_ratio):
                continue
            if normalize_text(t) in GENERIC_NORM:
                continue
            norm = normalize_text(t)
            if rel in EFFECT_RELATIONS | REACT_RELATIONS and norm.startswith("to "):
                continue
            if rel in ATTR_RELATIONS:
                if len(norm.split()) > 3:
                    continue
            if rel in REACT_RELATIONS and len(norm.split()) < rules.react_min_tokens:
                continue
            if jaccard_overlap(head, t) >= overlap:
                continue
            if t.lower() == "none":
                continue
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            rel_tails.append(t)
        if rel_tails:
            cleaned[rel] = rel_tails
    return cleaned


def generate_tails(
    heads: List[HeadResult],
    model_id: str,
    tp_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    base_tails: int,
    extra_tail: bool,
    overlap: float,
    batch_size: int,
    rules: QualityRules,
    base_temp: float,
    extra_temp: float,
) -> List[TailResult]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = make_llm(model_id, tp_size, max_model_len, gpu_mem_util)

    def run_batch(prompts: List[str], temperature: float, label: str) -> List[dict]:
        max_tokens = 512 if temperature <= 0.6 else 700
        params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens)
        parsed = []
        total = len(prompts)
        for start in range(0, total, batch_size):
            batch = prompts[start : start + batch_size]
            outputs = llm.generate(batch, params)
            for out in outputs:
                text = out.outputs[0].text
                obj = extract_json(text)
                parsed.append(obj or {})
            LOGGER.info("%s: %d/%d", label, min(start + batch_size, total), total)
        return parsed

    prompts = [
        format_chat(
            tokenizer,
            "You are generating training data. Return ONLY valid JSON. "
            "No extra text, bullets, numbering, or single-letter items.",
            build_tails_prompt(h.head, h.tags, base_tails),
        )
        for h in heads
    ]

    base_objs = run_batch(prompts, temperature=base_temp, label="Base tails")
    results: List[TailResult] = []

    good_heads: List[HeadResult] = []
    base_fail = 0
    for head, obj in zip(heads, base_objs):
        if not obj:
            base_fail += 1
            continue
        filtered = filter_tails(head.head, obj, min_tails=base_tails, overlap=overlap, rules=rules)
        if not filtered:
            base_fail += 1
            continue
        results.append(TailResult(head_result=head, tails=filtered))
        good_heads.append(head)

    LOGGER.info(
        "Base tails kept: %d/%d (failed %d)",
        len(results),
        len(heads),
        base_fail,
    )

    if extra_tail and good_heads:
        extra_prompts = [
            format_chat(
                tokenizer,
                "You add one more tail per relation. Return ONLY valid JSON. "
                "No extra text, bullets, numbering, or single-letter items.",
                build_tails_prompt(h.head, h.tags, 1),
            )
            for h in good_heads
        ]
        extra_objs = run_batch(extra_prompts, temperature=extra_temp, label="Extra tails")
        by_key = {hr.head_result.head.lower(): hr for hr in results}
        for idx, obj in enumerate(extra_objs):
            if not obj:
                continue
            filtered = filter_tails_partial(good_heads[idx].head, obj, overlap=overlap, rules=rules)
            if not filtered:
                continue
            # merge
            key = good_heads[idx].head.lower()
            target = by_key[key].tails
            for rel, rel_tails in filtered.items():
                for t in rel_tails[:1]:
                    if t not in target[rel]:
                        target[rel].append(t)

    del llm
    gc.collect()
    return results


def write_heads(path: str, heads: List[HeadResult]):
    with open(path, "w", encoding="utf-8") as f:
        for item in heads:
            f.write(
                json.dumps(
                    {
                        "head": item.head,
                        "tags": item.tags,
                        "sample_id": item.example.sample_id,
                        "country": item.example.country,
                        "region": item.example.region,
                        "sub_topic": item.example.sub_topic,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_heads(path: str) -> List[HeadResult]:
    heads: List[HeadResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ex = Example(
                sample_id=obj.get("sample_id", ""),
                country=obj.get("country", ""),
                region=obj.get("region", ""),
                sub_topic=obj.get("sub_topic", ""),
                scenario="",
            )
            heads.append(HeadResult(example=ex, head=obj["head"], tags=obj.get("tags", "")))
    return heads


def dedupe_by_head(items: List[TailResult]) -> List[TailResult]:
    seen = set()
    out = []
    for item in items:
        key = item.head_result.head.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def split_data(items: List[TailResult], val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    val_n = int(n * val_frac)
    test_n = int(n * test_frac)
    test = items[:test_n]
    val = items[test_n : test_n + val_n]
    train = items[test_n + val_n :]
    return train, val, test


def write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_atomic_bart_split(path_prefix: str, items: List[TailResult]):
    src_path = path_prefix + ".source"
    tgt_path = path_prefix + ".target"
    with open(src_path, "w", encoding="utf-8") as src_f, open(
        tgt_path, "w", encoding="utf-8"
    ) as tgt_f:
        for item in items:
            tags = item.head_result.tags
            head = item.head_result.head
            for rel in RELATIONS:
                for tail in item.tails[rel]:
                    prefix = f"{tags} " if tags else ""
                    src_f.write(f"{prefix}{head} {rel}\n")
                    tgt_f.write(f"{tail}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--head-model", required=True)
    parser.add_argument("--tail-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--keep-country-yes", action="store_true")
    parser.add_argument("--keep-discard-no", action="store_true")
    parser.add_argument("--add-region-tag", action="store_true")
    parser.add_argument("--add-country-tag", action="store_true")
    parser.add_argument("--base-tails", type=int, default=2)
    parser.add_argument("--add-third-tail", action="store_true")
    parser.add_argument("--overlap-threshold", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument("--tails-only", action="store_true")
    parser.add_argument("--heads-file", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-head-words", type=int, default=16)
    parser.add_argument("--quality-mode", choices=["standard", "strict"], default="standard")
    parser.add_argument("--tail-temp-base", type=float, default=0.6)
    parser.add_argument("--tail-temp-extra", type=float, default=0.7)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    LOGGER.info("Starting pipeline")
    LOGGER.info("Heads only: %s, Tails only: %s", args.heads_only, args.tails_only)

    if args.tails_only and not args.heads_file:
        raise SystemExit("--tails-only requires --heads-file")

    if args.tails_only:
        heads = read_heads(args.heads_file)
    else:
        examples, stats = load_arabculture(
            args.countries, args.keep_country_yes, args.keep_discard_no
        )
        LOGGER.info("Filtered rows: %s", stats)
        heads = generate_heads(
            examples,
            model_id=args.head_model,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            add_region=args.add_region_tag,
            add_country=args.add_country_tag,
            temperature=0.6,
            batch_size=args.batch_size,
            max_head_words=args.max_head_words,
        )
        heads_file = args.heads_file or os.path.join(args.output_dir, "heads.jsonl")
        write_heads(heads_file, heads)

    if args.heads_only:
        summary = {
            "countries": args.countries,
            "heads": len(heads),
            "heads_file": args.heads_file or os.path.join(args.output_dir, "heads.jsonl"),
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return

    rules = get_quality_rules(args.quality_mode)
    tails = generate_tails(
        heads,
        model_id=args.tail_model,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        base_tails=args.base_tails,
        extra_tail=args.add_third_tail,
        overlap=args.overlap_threshold,
        batch_size=args.batch_size,
        rules=rules,
        base_temp=args.tail_temp_base,
        extra_temp=args.tail_temp_extra,
    )

    tails = dedupe_by_head(tails)

    train, val, test = split_data(tails, args.val_frac, args.test_frac, args.seed)

    # Save rich JSONL
    def to_row(item: TailResult) -> dict:
        row = {"event": item.head_result.head}
        row.update(item.tails)
        row["meta"] = {
            "sample_id": item.head_result.example.sample_id,
            "country": item.head_result.example.country,
            "region": item.head_result.example.region,
            "sub_topic": item.head_result.example.sub_topic,
            "tags": item.head_result.tags,
        }
        return row

    write_jsonl(os.path.join(args.output_dir, "atomic_full.train.jsonl"), [to_row(x) for x in train])
    write_jsonl(os.path.join(args.output_dir, "atomic_full.val.jsonl"), [to_row(x) for x in val])
    write_jsonl(os.path.join(args.output_dir, "atomic_full.test.jsonl"), [to_row(x) for x in test])

    # Write atomic-bart splits
    bart_dir = os.path.join(args.output_dir, "atomic_bart")
    os.makedirs(bart_dir, exist_ok=True)
    write_atomic_bart_split(os.path.join(bart_dir, "train"), train)
    write_atomic_bart_split(os.path.join(bart_dir, "val"), val)
    write_atomic_bart_split(os.path.join(bart_dir, "test"), test)

    summary = {
        "countries": args.countries,
        "heads": len(heads),
        "tails": len(tails),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
