# Methodology: GD-BART -> ED-BART -> GD-COMET -> ED-COMET + AIN

## Introduction and task overview

This work studies visual commonsense reasoning in culturally grounded contexts.
We combine a commonsense generator (COMET) with a vision-language model (AIN)
to answer multiple-choice questions about images.

### What is VCR?

Visual Commonsense Reasoning (VCR) is a benchmark that tests whether a model
can answer questions about an image with commonsense knowledge. Each example
includes:
- an image,
- a question,
- four answer options (A/B/C/D).

The model must choose the correct option. VCR also includes a rationale task,
but EG-VCR provides Q->A only.

Example (simplified):
- Question: "Why is [person1] smiling?"
- Options: A) "Because [person1] is angry" B) "Because [person1] received good news" C) ...

### What is ATOMIC?

ATOMIC2020 is a commonsense knowledge graph of if-then relations about events.
It uses relations such as:
- xIntent (intent of the subject)
- xNeed (preconditions)
- xEffect (effects on the subject)
- oReact (effects on others)

Example:
- Head: "PersonX gives PersonY a gift"
- xIntent: "to show appreciation"
- oReact: "feels happy"

### What is COMET?

COMET is a generative model trained on ATOMIC to produce tails for these
relations. Given a head event, COMET generates plausible inferences.

Example:
- Input: "PersonX gives PersonY a gift xIntent"
- Output: "to show appreciation"

### Why commonsense reasoning matters

Many visual questions cannot be answered from pixels alone. Cultural practices,
social norms, and implicit human intentions often determine the correct answer.
Commonsense reasoning adds this missing context, enabling the model to:
- infer likely intentions or reactions,
- disambiguate between plausible options,
- generalize beyond literal visual cues.

This document describes the full pipeline used in this project, from BART pretraining
to ED-COMET and the AIN-based VCR/EG-VCR inference pipeline. The goal is to
document each stage, inputs/outputs, and the exact scripts that implement them.

## 1) Pretraining BART on CANDLE (GD-BART)

Goal: train a general-domain BART (GD-BART) using Arabic CANDLE data with a
denoising objective (fairseq BART pretraining).

High-level steps:
- Prepare CANDLE data and binarize with fairseq (BPE and dict).
- Train BART from scratch using the denoising objective.
- Save the fairseq checkpoint (GD-BART).

Outputs:
- Fairseq checkpoint (GD-BART) used as the base for continued pretraining.
- Tokenizer/dict files (must remain consistent for later conversion).

Notes:
- This step is performed in the BART-pretraining pipeline (outside this repo).
- The checkpoint is later converted to HuggingFace format when using the
  Transformers-based COMET finetuning script.

## 2) Continued pretraining with upsampled Egypt data (ED-BART)

Goal: bias GD-BART toward Egyptian and MENA content by continuing pretraining
on an upsampled Egyptian subset.

High-level steps:
- Build a training mix that upweights Egyptian data.
- Continue BART denoising training from GD-BART weights.
- Save the new checkpoint (ED-BART).

Outputs:
- Fairseq checkpoint (ED-BART).
- Same tokenizer/dict as GD-BART to keep compatibility.

Notes:
- This step keeps the same vocabulary and tokenization as GD-BART.
- ED-BART is later used as the base for ED-COMET finetuning.

## 3) Finetune on ATOMIC -> GD-COMET

Goal: finetune BART on ATOMIC2020 to obtain a general-domain COMET model
(GD-COMET).

Steps and scripts in this repo:

1) Convert ATOMIC2020 to BART format:
```bash
python scripts/prepare_atomic_for_bart.py \
  --input_dir ./atomic2020_data-feb2021 \
  --output_dir ./data/atomic_bart
```

2) Convert fairseq BART to HuggingFace format (if using HF finetuning):
```bash
python -m transformers.models.bart.convert_bart_original_pytorch_checkpoint_to_pytorch \
  --fairseq_path /path/to/gd_bart_checkpoint.pt \
  --pytorch_dump_folder_path ./checkpoints/bart_hf
```

3) Finetune with the COMET training script:
```bash
CUDA_VISIBLE_DEVICES=0 python models/comet_atomic2020_bart/finetune.py \
  --task summarization \
  --do_train --do_predict \
  --data_dir ./data/atomic_bart \
  --output_dir ./results/bart_atomic_finetune \
  --model_name_or_path ./checkpoints/bart_hf \
  --atomic
```

Outputs:
- GD-COMET checkpoint (HuggingFace or fairseq, depending on the training path).

## 4) Finetune on Arab-ATOMIC -> ED-COMET

Goal: specialize COMET for MENA and Egyptian commonsense by finetuning on an
ArabCulture-derived ATOMIC-style dataset (Arab-ATOMIC).

This stage uses the ArabCulture pipeline in this repo to produce ATOMIC-style
head/tail pairs, then mixes them with ATOMIC for finetuning.

### 4.1 ArabCulture data generation pipeline

Entry point:
- `arabculture_pipeline/run_pipeline.py`

Core steps:
- Load ArabCulture (MSA MCQs).
- Filter to the target countries and quality flags.
- Build heads from the Arabic statement (Jais-2-8B-Chat).
- Generate tails for 9 relations (Qwen2.5-7B-Instruct).
- Add region/country tags such as `<REGION=MENA> <COUNTRY=EGY>`.
- Write `atomic_full.{train,val,test}.jsonl` and `atomic_bart/` splits.

Example command:
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

Optional quality controls:
- `arabculture_pipeline/filter_heads.py` for head cleanup.
- `arabculture_pipeline/repair_tails.py` for tail cleanup.
- `arabculture_pipeline/build_atomic_bart_from_jsonl.py` to rebuild .source/.target.

### 4.2 Build mixed Arab-ATOMIC dataset

Use the helper to combine ATOMIC and ArabCulture (15-30 percent mix):
```bash
bash arabculture_pipeline/build_atomic_mix.sh
```

This creates `data/arabculture_atomic_combined/atomic_bart` which can be
used for finetuning.

### 4.3 Finetune ED-COMET

Finetune the COMET model on the mixed dataset using the same training
recipe as in step 3, but with the Arab-ATOMIC data.

Output:
- ED-COMET checkpoint, used for inference on EG-VCR and VCR.

## 5) AIN + ED-COMET VCR/EG-VCR inference pipeline

This pipeline combines:
- AIN for image-grounded answer selection.
- ED-COMET for commonsense augmentation.
- A head generation step to align visual content with COMET inputs.

End-to-end flow (high level):
1) Convert dataset to VCR-style JSONL with local image paths.
2) Generate a `PersonX ...` head from each image (AIN caption + entity parsing).
3) Generate ED-COMET tails for each head and relation.
4) Select and inject the most relevant tails into each option prompt.
5) Score answer choices with AIN and choose the best option (A-D).

### 5.1 Convert EG-VCR to JSONL

EG-VCR on HF uses only the `train` split (104 examples). Convert it to
VCR-style JSONL with local image paths:
```bash
python vcr_ain/egvcr_to_vcrjsonl.py \
  --split train \
  --output /path/to/egvcr_qa.jsonl \
  --images-dir /path/to/egvcr_images \
  --draw-boxes
```

### 5.2 Generate heads (AIN caption + entities)

AIN generates a caption and entities, then converts the caption to a
`PersonX ...` head:
```bash
python vcr_ain/build_heads_from_ain.py \
  --input /path/to/egvcr_qa.jsonl \
  --output /path/to/egvcr_qa.heads.jsonl \
  --model MBZUAI/AIN
```

### 5.3 ED-COMET cache (fairseq)

Generate COMET tails for each head and relation:
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

### 5.4 Inject ED-COMET into the prompt

This step selects the most relevant tails for each answer option and builds
an `augmented_prompt`.

Scoring and selection:
- For each option, build a query string: `question + option`.
- Score each candidate tail against the query.
- Select top-k tails per option and append them to the prompt.

We use SBERT similarity by default:
- Encode the query and each tail with a sentence-transformer.
- Compute cosine similarity.
- Keep the highest-scoring tails.

Fallback:
- If SBERT is not available, use Jaccard overlap over tokens.

Noise control:
- Drop `none` and `<unk>` tails.
- Optional uniform-tail gating: if all options receive identical tails,
  skip ED-COMET injection for that example.

Recommended settings (SBERT + filtering + uniform-tail gating):
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

### 5.5 AIN inference (baseline or ED-COMET)

AIN uses the image and the prompt and outputs a single letter A/B/C/D.

We evaluate with a scoring-based decoding method:
- For each candidate letter (A-D), append the letter token to the prompt.
- Compute the model loss for each candidate.
- Choose the letter with the lowest loss (highest log-likelihood).

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

Optional LoRA adapter:
```bash
--lora /path/to/ain_lora_qa/checkpoint-1000
```

### 5.6 LoRA finetuning for AIN (Q->A then QA->R)

Stage 1 (Q->A):
```bash
python vcr_ain/train_ain_lora.py \
  --train-jsonl /path/to/vcr_qa.sft.jsonl \
  --valid-jsonl /path/to/vcr_qa_val.sft.jsonl \
  --output-dir /path/to/ain_lora_qa \
  --model MBZUAI/AIN \
  --input-format qwen \
  --bf16
```

Stage 2 (QA->R):
```bash
python vcr_ain/train_ain_lora.py \
  --train-jsonl /path/to/vcr_qar.sft.jsonl \
  --valid-jsonl /path/to/vcr_qar_val.sft.jsonl \
  --output-dir /path/to/ain_lora_qar \
  --model MBZUAI/AIN \
  --input-format qwen \
  --bf16
```

## 6) Evaluation

Metric:
- Accuracy on Q->A (EG-VCR only provides Q->A).

The `ain_infer_vcr.py` script computes accuracy when targets are present.

## 7) Experimental setup

Datasets:
- CANDLE (BART pretraining)
- ATOMIC2020
- ARAB-ATOMIC (ArabCulture-derived ATOMIC-style data)
- EC-VCR (EG-VCR Q->A split)

Models:
- GD-BART (BART pretrained on CANDLE)
- ED-BART (continued pretraining with upsampled Egypt data)
- GD-COMET (finetuned on ATOMIC2020)
- ED-COMET (finetuned on ARAB-ATOMIC + ATOMIC mix)
- AIN (Qwen2-VL based, optional LoRA for VCR QA)

Inference configuration:
- ED-COMET cache generated with fairseq checkpoint.
- Relations: xIntent, xNeed, xReact, xEffect, oReact, oEffect.
- ED-COMET injection with SBERT scorer, tail filtering, and uniform-tail gating.
- AIN inference in score mode for stable option selection.

## 8) Main findings

- AIN baseline accuracy on EC-VCR Q->A: 0.61.
- AIN + ED-COMET improves accuracy (best run: 0.75).
- Gated ED-COMET injection reduces noise and improves stability.

## 9) Main libraries and tools used

Core ML stack:
- PyTorch
- Transformers
- Accelerate
- PEFT (LoRA)
- Fairseq (BART pretraining and COMET)

Vision and multimodal:
- Qwen2-VL (AIN model)
- qwen-vl-utils
- PIL (image handling)

Commonsense:
- sentence-transformers (SBERT matching)

Data processing:
- datasets (HF datasets)
- vllm (ArabCulture pipeline head/tail generation)

Support scripts:
- `scripts/prepare_atomic_for_bart.py`
- `scripts/finetune_bart_on_atomic.sh`
- `scripts/finetune_bart_on_atomic_fairseq.sh`
- `arabculture_pipeline/run_pipeline.py`
- `vcr_ain/*.py` (VCR/EG-VCR pipeline)
