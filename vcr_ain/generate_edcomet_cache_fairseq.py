#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
from typing import Dict, List, Tuple


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ED-COMET cache using fairseq BART.")
    parser.add_argument("--input", required=True, help="JSONL with `head` and optional `tags`.")
    parser.add_argument("--output", required=True, help="Output cache JSONL.")
    parser.add_argument("--model-dir", required=True, help="Directory containing checkpoint.")
    parser.add_argument("--checkpoint-file", required=True, help="Checkpoint filename (.pt).")
    parser.add_argument("--data-bin", required=True, help="Fairseq data-bin with dict.")
    parser.add_argument(
        "--fairseq-interactive",
        default="fairseq-interactive",
        help="Path to fairseq-interactive binary.",
    )
    parser.add_argument("--tags", default="<REGION=MENA> <COUNTRY=EGY>")
    parser.add_argument(
        "--relations",
        default="xIntent,xNeed,xReact,xEffect,oReact,oEffect",
        help="Comma-separated relations.",
    )
    parser.add_argument("--num-tails", type=int, default=2)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--max-len-b", type=int, default=48)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    relations = [r.strip() for r in args.relations.split(",") if r.strip()]
    ckpt_path = f"{args.model_dir.rstrip('/')}/{args.checkpoint_file}"

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        batch: List[Dict[str, str]] = []
        for line in fin:
            rec = json.loads(line)
            head = rec.get("head", "").strip()
            if not head:
                continue
            tags = rec.get("tags", args.tags)
            batch.append({"head": head, "tags": tags})

            if len(batch) >= args.batch_size:
                _write_batch(batch, relations, args, fout, ckpt_path)
                batch = []

        if batch:
            _write_batch(batch, relations, args, fout, ckpt_path)


def _run_fairseq_interactive(
    prompts: List[str],
    args,
    ckpt_path: str,
) -> List[str]:
    if not prompts:
        return []
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        for p in prompts:
            tmp.write(p + "\n")
        input_path = tmp.name

    cmd = [
        args.fairseq_interactive,
        args.data_bin,
        "--path",
        ckpt_path,
        "--source-lang",
        "source",
        "--target-lang",
        "target",
        "--input",
        input_path,
        "--beam",
        str(args.beam),
        "--max-len-b",
        str(args.max_len_b),
        "--max-len-a",
        "0",
        "--min-len",
        str(args.min_len),
        "--batch-size",
        str(args.batch_size),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    hyps: Dict[int, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("H-"):
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            idx = int(parts[0][2:])
            hyps[idx] = parts[2]

    outputs = []
    for i in range(len(prompts)):
        outputs.append(hyps.get(i, ""))
    return outputs


def _write_batch(batch, relations, args, fout, ckpt_path):
    jobs: List[Tuple[int, str, str]] = []
    for idx, item in enumerate(batch):
        head = item["head"]
        tags = item["tags"]
        for rel in relations:
            prompt = f"{tags} {head} {rel}"
            for _ in range(args.num_tails):
                jobs.append((idx, rel, prompt))

    outputs = _run_fairseq_interactive([j[2] for j in jobs], args, ckpt_path)
    grouped: Dict[Tuple[int, str], List[str]] = {}
    for (idx, rel, _), tail in zip(jobs, outputs):
        grouped.setdefault((idx, rel), []).append(tail)

    for idx, item in enumerate(batch):
        head = item["head"]
        tags = item["tags"]
        out: Dict[str, List[str]] = {"event": head, "meta": {"tags": tags}}
        for rel in relations:
            tails = grouped.get((idx, rel), [])
            seen = set()
            unique = []
            for t in tails:
                t = t.strip()
                if not t:
                    continue
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(t)
            out[rel] = unique
        fout.write(json.dumps(out, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
