#!/usr/bin/env python3
import argparse
import json
import logging
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception:
    SentenceTransformer = None
    util = None


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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def jaccard(a: str, b: str) -> float:
    a_set = set(normalize(a).split())
    b_set = set(normalize(b).split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def load_comet_cache(path: str, relations: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    rels = set(relations)
    cache: Dict[str, Dict[str, List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            head = obj.get("event") or obj.get("head") or obj.get("head_event")
            if not head:
                continue
            head_key = normalize(head)
            tails: Dict[str, List[str]] = {}
            for rel in rels:
                vals = obj.get(rel, [])
                if vals:
                    tails[rel] = list(vals)
            if tails:
                cache[head_key] = tails
    return cache


def is_none_tail(text: str) -> bool:
    return normalize(text) in {"none", "n/a", "na", "unknown", "unk"}


def collect_candidates(
    tails_by_rel: Dict[str, List[str]],
    drop_none: bool,
) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for rel, tails in tails_by_rel.items():
        for t in tails:
            t = t.strip()
            if not t:
                continue
            if drop_none and is_none_tail(t):
                continue
            if t:
                candidates.append((rel, t))
    return candidates


def score_candidates(
    query: str,
    candidates: List[Tuple[str, str]],
    scorer: str,
    sbert_model: Optional[SentenceTransformer] = None,
) -> List[Tuple[float, Tuple[str, str]]]:
    if scorer == "sbert" and sbert_model and util:
        texts = [query] + [c[1] for c in candidates]
        embs = sbert_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        q_emb = embs[0]
        c_embs = embs[1:]
        sims = util.cos_sim(q_emb, c_embs).squeeze(0).tolist()
        return list(zip(sims, candidates))
    scored = [(jaccard(query, c[1]), c) for c in candidates]
    return scored


def select_topk(
    query: str,
    candidates: List[Tuple[str, str]],
    k: int,
    scorer: str,
    sbert_model: Optional[SentenceTransformer],
) -> List[Tuple[str, str]]:
    if not candidates or k <= 0:
        return []
    scored = score_candidates(query, candidates, scorer, sbert_model)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def format_tail(rel: str, tail: str) -> str:
    return f"{rel}: {tail}"


def build_augmented_prompt(
    base_prompt: str,
    selected_by_option: Dict[str, List[Tuple[str, str]]],
    tags: str,
) -> str:
    lines: List[str] = [base_prompt]
    if tags:
        lines.append(f"Tags: {tags}")
    if any(selected_by_option.values()):
        lines.append("Useful commonsense:")
        for letter in sorted(selected_by_option.keys()):
            tails = selected_by_option[letter]
            if not tails:
                continue
            formatted = " | ".join(format_tail(rel, t) for rel, t in tails)
            lines.append(f"{letter}) {formatted}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject ED-COMET inferences into VCR prompts.")
    parser.add_argument("--input", required=True, help="Input JSONL (from preprocess_vcr).")
    parser.add_argument("--output", required=True, help="Output JSONL with augmented_prompt.")
    parser.add_argument("--comet-cache", required=True, help="JSONL with ED-COMET outputs.")
    parser.add_argument("--k", type=int, default=5, help="Top-k inferences per option.")
    parser.add_argument(
        "--relations",
        default="xIntent,xNeed,xReact,xEffect,oReact,oEffect",
        help="Comma-separated relations to use.",
    )
    parser.add_argument("--tags", default="<REGION=MENA> <COUNTRY=EGY>")
    parser.add_argument("--scorer", choices=["jaccard", "sbert"], default="jaccard")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--keep-none",
        action="store_true",
        help="Keep COMET 'none' tails instead of filtering them.",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=1,
        help="Min total candidate tails required to inject ED-COMET.",
    )
    args = parser.parse_args()

    relations = [r.strip() for r in args.relations.split(",") if r.strip()]
    comet_cache = load_comet_cache(args.comet_cache, relations)

    sbert_model = None
    if args.scorer == "sbert":
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available; falling back to jaccard.")
        else:
            sbert_model = SentenceTransformer(args.sbert_model)

    total = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            head = normalize(rec.get("head", ""))
            tails_by_rel = comet_cache.get(head, {})
            candidates = collect_candidates(tails_by_rel, drop_none=not args.keep_none)

            selected_by_option: Dict[str, List[Tuple[str, str]]] = {}
            if len(candidates) >= args.min_candidates:
                for letter, opt_text in rec.get("options", {}).items():
                    query = f"{rec.get('question', '')} {opt_text}"
                    selected_by_option[letter] = select_topk(
                        query, candidates, args.k, args.scorer, sbert_model
                    )
            else:
                for letter in rec.get("options", {}).keys():
                    selected_by_option[letter] = []

            rec["augmented_prompt"] = build_augmented_prompt(
                rec.get("prompt", ""), selected_by_option, args.tags
            )
            rec["selected_inferences"] = {
                letter: [format_tail(rel, t) for rel, t in tails]
                for letter, tails in selected_by_option.items()
            }
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            total += 1

    print(f"Wrote {total} augmented records to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
