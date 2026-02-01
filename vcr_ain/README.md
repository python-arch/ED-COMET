# VCR + AIN + ED-COMET (starter pipeline)

This folder contains a minimal, runnable preprocessing + injection scaffold for the plan:

1) preprocess VCR JSONL into AIN-friendly JSONL  
2) add naive `PersonX ...` heads (optional)  
3) inject ED-COMET inferences into prompts

## 1) Preprocess VCR

```bash
python vcr_ain/preprocess_vcr.py \
  --vcr-jsonl /path/to/vcr1/train.jsonl \
  --images-dir /path/to/vcr1images \
  --out-dir /path/to/out \
  --include-rationale
```

Outputs:

- `vcr_qa.jsonl` (Q→A)
- `vcr_qar.jsonl` (QA→R)

## 2) Add heads (AIN caption + entities)

This uses AIN to caption the image, extract entities, and build a PersonX head.

```bash
python vcr_ain/build_heads_from_ain.py \
  --input /path/to/out/vcr_qa.jsonl \
  --output /path/to/out/vcr_qa.heads.jsonl \
  --model MBZUAI/AIN
```

If you want the naive question-based head builder instead:

```bash
python vcr_ain/build_heads_from_questions.py \
  --input /path/to/out/vcr_qa.jsonl \
  --output /path/to/out/vcr_qa.heads.jsonl
```

## 3) Generate ED-COMET cache (fairseq .pt or HF)

Fairseq-native (uses your .pt directly):

```bash
python vcr_ain/generate_edcomet_cache_fairseq.py \
  --input /path/to/out/vcr_qa.heads.jsonl \
  --output /path/to/out/edcomet_cache.jsonl \
  --model-dir /path/to/checkpoints/comet_finetune_arabculture_mix \
  --checkpoint-file checkpoint_last.pt \
  --data-bin /path/to/data-bin/atomic_mix_arabculture_30 \
  --relations xIntent,xNeed,xReact,xEffect,oReact,oEffect \
  --num-tails 2
```

HF model (if you converted):

```bash
python vcr_ain/generate_edcomet_cache.py \
  --input /path/to/out/vcr_qa.heads.jsonl \
  --output /path/to/out/edcomet_cache.jsonl \
  --model /path/to/edcomet-hf \
  --relations xIntent,xNeed,xReact,xEffect,oReact,oEffect \
  --num-tails 2
```

## 4) Inject ED-COMET inferences

This expects a precomputed ED-COMET cache JSONL (one record per head):

```json
{"event": "PersonX ...", "xIntent": ["..."], "xNeed": ["..."], ...}
```

```bash
python vcr_ain/inject_edcomet.py \
  --input /path/to/out/vcr_qa.heads.jsonl \
  --output /path/to/out/vcr_qa.aug.jsonl \
  --comet-cache /path/to/ed_comet_cache.jsonl \
  --relations xIntent,xNeed,xReact,xEffect,oReact,oEffect \
  --k 5 \
  --scorer jaccard
```

If you want SBERT scoring, install `sentence-transformers` and use:

```bash
--scorer sbert --sbert-model all-MiniLM-L6-v2
```

## 5) Prepare AIN SFT data

```bash
python vcr_ain/prepare_ain_sft.py \
  --input /path/to/out/vcr_qa.aug.jsonl \
  --output /path/to/out/vcr_qa.sft.jsonl \
  --format qwen
```

## 6) Run AIN inference (baseline or +ED-COMET)

```bash
python vcr_ain/ain_infer_vcr.py \
  --input /path/to/out/vcr_qa.aug.jsonl \
  --output /path/to/out/vcr_qa.preds.jsonl \
  --model MBZUAI/AIN \
  --mode generate
```

## Next

This scaffold does not yet run AIN itself. It gives you clean inputs and
augmented prompts so you can plug in your AIN inference/training loop.
