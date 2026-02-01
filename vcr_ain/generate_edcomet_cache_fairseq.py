#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List


def load_bart(model_dir: str, checkpoint_file: str, data_bin: str):
    from fairseq.models.bart import BARTModel

    bart = BARTModel.from_pretrained(
        model_dir, checkpoint_file=checkpoint_file, data_name_or_path=data_bin
    )
    bart.eval()
    if bart.device.type != "cpu":
        return bart
    return bart.cuda() if bart.is_cuda_available() else bart


def generate_tails(bart, prompts: List[str], beam: int, max_len_b: int, min_len: int) -> List[str]:
    return bart.sample(prompts, beam=beam, max_len_b=max_len_b, min_len=min_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ED-COMET cache using fairseq BART.")
    parser.add_argument("--input", required=True, help="JSONL with `head` and optional `tags`.")
    parser.add_argument("--output", required=True, help="Output cache JSONL.")
    parser.add_argument("--model-dir", required=True, help="Directory containing checkpoint.")
    parser.add_argument("--checkpoint-file", required=True, help="Checkpoint filename (.pt).")
    parser.add_argument("--data-bin", required=True, help="Fairseq data-bin with dict.")
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
    bart = load_bart(args.model_dir, args.checkpoint_file, args.data_bin)

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
                _write_batch(bart, batch, relations, args, fout)
                batch = []

        if batch:
            _write_batch(bart, batch, relations, args, fout)


def _write_batch(bart, batch, relations, args, fout):
    for item in batch:
        head = item["head"]
        tags = item["tags"]
        out: Dict[str, List[str]] = {"event": head, "meta": {"tags": tags}}
        for rel in relations:
            prompt = f"{tags} {head} {rel}"
            prompts = [prompt] * args.num_tails
            tails = generate_tails(
                bart,
                prompts,
                beam=args.beam,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
            )
            out[rel] = [t.strip() for t in tails if t.strip()]
        fout.write(json.dumps(out, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
