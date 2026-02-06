# ED-COMET + AIN: Culturally Grounded Visual Commonsense Reasoning

This repo implements a full pipeline for culturally grounded visual commonsense
reasoning, combining a COMET-style commonsense generator (ED-COMET) with a
vision-language answer ranker (AIN). It includes:

- BART curriculum for commonsense generation (GD-BART -> ED-BART -> GD-COMET -> ED-COMET)
- ArabCulture-to-ATOMIC dataset generation (Arab-ATOMIC)
- VCR/EG-VCR preprocessing and head generation
- ED-COMET cache generation and prompt injection
- AIN inference and optional LoRA finetuning

## 1) Training curriculum (core methodology)

We use a staged curriculum to gradually introduce culturally grounded commonsense:

1) **GD-BART pretraining (CANDLE)**  
   Pretrain BART with a denoising objective on CANDLE for general Arabic priors.

2) **ED-BART continued pretraining (Egypt upsampling)**  
   Continue pretraining from GD-BART, upweighting Egyptian data (e.g., x5).

3) **GD-COMET finetuning (ATOMIC2020)**  
   Finetune on ATOMIC to learn causal/social relations:
   xIntent, xNeed, xReact, xEffect, oReact, oEffect.

4) **ED-COMET specialization (ARAB-ATOMIC + ATOMIC mix)**  
   Further finetune on a mix of Arab-ATOMIC and ATOMIC (lambda 0.15-0.30).

Notes:
- Tokenizer/dict must remain consistent across all stages.
- ED-COMET uses region/country tags (e.g., `<REGION=MENA> <COUNTRY=EGY>`).

## 2) ArabCulture -> Arab-ATOMIC generation pipeline

This pipeline generates an ATOMIC-style dataset from ArabCulture MCQs.
Entry point: `arabculture_pipeline/run_pipeline.py`.

Example:

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
  --add-third-tail
```

Outputs:
- `atomic_full.{train,val,test}.jsonl`
- `atomic_bart/{train,val,test}.{source,target}`

Optional cleanup:
- `arabculture_pipeline/filter_heads.py`
- `arabculture_pipeline/repair_tails.py`
- `arabculture_pipeline/build_atomic_bart_from_jsonl.py`

## 3) GD-COMET finetuning (ATOMIC)

Prepare ATOMIC in BART format:

```bash
python scripts/prepare_atomic_for_bart.py \
  --input_dir ./atomic2020_data-feb2021 \
  --output_dir ./data/atomic_bart
```

Finetune (Transformers script):

```bash
CUDA_VISIBLE_DEVICES=0 python models/comet_atomic2020_bart/finetune.py \
  --task summarization \
  --do_train --do_predict \
  --data_dir ./data/atomic_bart \
  --output_dir ./results/bart_atomic_finetune \
  --model_name_or_path ./checkpoints/bart_hf \
  --atomic
```

## 4) ED-COMET finetuning (Arab-ATOMIC + ATOMIC mix)

Create the mixed dataset (example script):

```bash
bash arabculture_pipeline/build_atomic_mix.sh
```

Then finetune with the same recipe as GD-COMET, but using the mixed data.

## 5) VCR/EG-VCR + AIN + ED-COMET inference pipeline

### 5.1 Convert EG-VCR to JSONL (Q->A)

```bash
python vcr_ain/egvcr_to_vcrjsonl.py \
  --split train \
  --output /path/to/egvcr_qa.jsonl \
  --images-dir /path/to/egvcr_images \
  --draw-boxes
```

### 5.2 Generate heads (AIN caption + entities -> PersonX)

```bash
python vcr_ain/build_heads_from_ain.py \
  --input /path/to/egvcr_qa.jsonl \
  --output /path/to/egvcr_qa.heads.jsonl \
  --model MBZUAI/AIN
```

### 5.3 ED-COMET cache (fairseq)

```bash
python vcr_ain/generate_edcomet_cache_fairseq.py \
  --input /path/to/egvcr_qa.heads.jsonl \
  --output /path/to/egvcr_edcomet_cache.jsonl \
  --model-dir /path/to/checkpoints/comet_finetune_arabculture_mix \
  --checkpoint-file checkpoint_last.pt \
  --data-bin /path/to/data-bin/atomic_mix_arabculture_30 \
  --relations xIntent,xNeed,xReact,xEffect,oReact,oEffect \
  --num-tails 5 \
  --beam 10 \
  --max-len-b 64
```

### 5.4 Inject ED-COMET into prompts

```bash
python vcr_ain/inject_edcomet.py \
  --input /path/to/egvcr_qa.heads.jsonl \
  --output /path/to/egvcr_qa.edcomet.jsonl \
  --comet-cache /path/to/egvcr_edcomet_cache.jsonl \
  --k 5 \
  --scorer sbert \
  --min-candidates 3 \
  --drop-uniform
```

### 5.5 AIN inference

Baseline:

```bash
python vcr_ain/ain_infer_vcr.py \
  --input /path/to/egvcr_qa.heads.jsonl \
  --output /path/to/egvcr_qa.base.pred.jsonl \
  --model MBZUAI/AIN \
  --mode score
```

ED-COMET:

```bash
python vcr_ain/ain_infer_vcr.py \
  --input /path/to/egvcr_qa.edcomet.jsonl \
  --output /path/to/egvcr_qa.edcomet.pred.jsonl \
  --model MBZUAI/AIN \
  --mode score
```

Optional LoRA:

```bash
--lora /path/to/ain_lora_qa/checkpoint-1000
```

## 6) AIN LoRA finetuning (Q->A, then QA->R)

```bash
python vcr_ain/train_ain_lora.py \
  --train-jsonl /path/to/vcr_qa.sft.jsonl \
  --valid-jsonl /path/to/vcr_qa_val.sft.jsonl \
  --output-dir /path/to/ain_lora_qa \
  --model MBZUAI/AIN \
  --input-format qwen \
  --bf16
```

## 7) Main libraries and tools

- PyTorch
- Transformers
- Accelerate
- PEFT (LoRA)
- Fairseq
- sentence-transformers (SBERT)
- datasets (HF)
- vllm
- qwen-vl-utils

## 8) Results

- AIN baseline on EG-VCR Q->A: 0.61
- AIN + ED-COMET on EG-VCR Q->A: 0.75

## 9) Notes and gotchas

- Dict/tokenizer sizes must match the COMET checkpoint.
- For fairseq COMET, keep dict files at the original size (fairseq adds 4 special tokens).
- Use gating (drop uniform tails) to avoid noisy injections.
