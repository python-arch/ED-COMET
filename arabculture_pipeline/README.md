# ArabCulture → ATOMIC pipeline (Jais heads + Qwen tails)

This pipeline builds an **ATOMIC-style English dataset** from ArabCulture (MSA Arabic MCQs) using:

- **Jais-2-8B-Chat** for head generation (Arabic → English `PersonX ...`)
- **Qwen2.5-7B-Instruct** for tails (9 relations, 2–3 tails each)

It outputs:
- `atomic_full.{train,val,test}.jsonl` (one JSON per head with 9 relation arrays)
- `atomic_bart/{train,val,test}.{source,target}` (ready for your ATOMIC-BART finetuning)
- `summary.json` (counts)

## Requirements

- Python 3.10+
- `vllm`
- `datasets`
- `transformers`

Example install:

```bash
pip install vllm datasets transformers
```

> Note: `datasets` downloads ArabCulture from Hugging Face. You need network access in the runtime environment.

## Dataset config (per your plan)

Countries to include (8 total):

```
Egypt Jordan Palestine KSA UAE Morocco Tunisia Sudan
```

Filters:
- keep `should_discard == "No"`
- keep `relevant_to_this_country == "Yes"`

Head construction:
- `first_statement + correct_option`

Tags:
- source prefix: `<REGION=MENA> <COUNTRY=XXX>`

## Run command (single GPU)

```bash
python arabculture_pipeline/run_pipeline.py \
  --countries Egypt Jordan Palestine KSA UAE Morocco Tunisia Sudan \
  --head-model inceptionai/Jais-2-8B-Chat \
  --tail-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir /path/to/output/arabculture_atomic \
  --keep-country-yes \
  --keep-discard-no \
  --add-region-tag \
  --add-country-tag \
  --base-tails 2 \
  --add-third-tail \
  --tp-size 1
```

## Split runs (avoid vLLM OOM between models)

If vLLM holds onto GPU memory between models, run heads and tails separately.

Heads only:

```bash
python arabculture_pipeline/run_pipeline.py \
  --countries Egypt Jordan Palestine KSA UAE Morocco Tunisia Sudan \
  --head-model inceptionai/Jais-2-8B-Chat \
  --tail-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir /path/to/output/arabculture_atomic \
  --keep-country-yes \
  --keep-discard-no \
  --add-region-tag \
  --add-country-tag \
  --base-tails 2 \
  --add-third-tail \
  --heads-only
```

Tails only (uses heads.jsonl produced above):

```bash
python arabculture_pipeline/run_pipeline.py \
  --countries Egypt Jordan Palestine KSA UAE Morocco Tunisia Sudan \
  --head-model inceptionai/Jais-2-8B-Chat \
  --tail-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir /path/to/output/arabculture_atomic \
  --heads-file /path/to/output/arabculture_atomic/heads.jsonl \
  --base-tails 2 \
  --add-third-tail \
  --tails-only
```

## Multi-GPU (tensor parallel)

If you want to use multiple A100s, set `--tp-size`:

```bash
--tp-size 2
```

This uses vLLM tensor parallel.

## Outputs

```
output_dir/
  summary.json
  atomic_full.train.jsonl
  atomic_full.val.jsonl
  atomic_full.test.jsonl
  atomic_bart/
    train.source
    train.target
    val.source
    val.target
    test.source
    test.target
```

## Notes on quality

- The script filters generic tails and head-copying.
- Tails are constrained to be short, distinct, and non-generic.
- Use `summary.json` to confirm total counts.
- You can inspect `atomic_full.*.jsonl` for manual sanity checks.

## Suggested next steps

1) Merge `atomic_bart/` outputs with your existing ATOMIC mix (15–30% mix).
2) Finetune COMET-BART with tags on the synthetic set.
3) Evaluate with your gold Egyptian set (86 events).

If you want a LoRA-only training script or a merging script, I can add it.

## Logging + progress

The pipeline writes logs to:

```
output_dir/run.log
```

Progress is reported per batch. You can change the batch size:

```
--batch-size 256
```

## Quality modes

Use strict filtering + relation rules for higher-quality tails:

```bash
--quality-mode strict
```

You can also lower the sampling temperature for cleaner generations:

```bash
--tail-temp-base 0.5 --tail-temp-extra 0.6
```

## Head filtering (optional)

Remove noisy heads (slashes, low alpha ratio) before tails:

```bash
python arabculture_pipeline/filter_heads.py \
  --input /path/to/output/arabculture_atomic/heads.jsonl \
  --output /path/to/output/arabculture_atomic/heads.filtered.jsonl \
  --min-alpha-ratio 0.6 \
  --max-words 16 \
  --reject-other-persons \
  --reject-arabic \
  --reject-personx-article \
  --reject-adverbs \
  --reject-demonyms
```

Then run tails-only with `--heads-file heads.filtered.jsonl`.

## Tail repair (optional)

Light spelling/grammar correction for suspicious tails only:

```bash
python arabculture_pipeline/repair_tails.py \
  --input /path/to/output/arabculture_atomic/atomic_full.train.jsonl \
  --output /path/to/output/arabculture_atomic/atomic_full.train.repaired.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tp-size 1
```

Repeat for val/test if desired, then rebuild `.source/.target` from the repaired JSONL.
