#!/usr/bin/env bash
set -euo pipefail

VCR_ROOT="/home/ahmedjaheen/data/vcr1"
OUT_TRAIN="/home/ahmedjaheen/data/vcr1_out"
OUT_VAL="/home/ahmedjaheen/data/vcr1_out_val"
AIN_MODEL="MBZUAI/AIN"
MAX_LENGTH=4096

# 1) Preprocess train + val
python vcr_ain/preprocess_vcr.py \
  --vcr-jsonl "${VCR_ROOT}/train.jsonl" \
  --images-dir "${VCR_ROOT}/vcr1images" \
  --out-dir "${OUT_TRAIN}" \
  --include-rationale

python vcr_ain/preprocess_vcr.py \
  --vcr-jsonl "${VCR_ROOT}/val.jsonl" \
  --images-dir "${VCR_ROOT}/vcr1images" \
  --out-dir "${OUT_VAL}" \
  --include-rationale

# 2) Build SFT JSONL for Q→A
python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_TRAIN}/vcr_qa.jsonl" \
  --output "${OUT_TRAIN}/vcr_qa.sft.jsonl" \
  --format qwen

python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_VAL}/vcr_qa.jsonl" \
  --output "${OUT_VAL}/vcr_qa.sft.jsonl" \
  --format qwen

# 3) Stage 1: Q→A LoRA finetune
python vcr_ain/train_ain_lora.py \
  --train-jsonl "${OUT_TRAIN}/vcr_qa.sft.jsonl" \
  --valid-jsonl "${OUT_VAL}/vcr_qa.sft.jsonl" \
  --output-dir "/home/ahmedjaheen/data/ain_lora_qa" \
  --model "${AIN_MODEL}" \
  --input-format qwen \
  --max-length "${MAX_LENGTH}" \
  --bf16 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 2e-5 \
  --num-epochs 2

# 4) Build SFT JSONL for QA→R
python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_TRAIN}/vcr_qar.jsonl" \
  --output "${OUT_TRAIN}/vcr_qar.sft.jsonl" \
  --format qwen

python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_VAL}/vcr_qar.jsonl" \
  --output "${OUT_VAL}/vcr_qar.sft.jsonl" \
  --format qwen

# 5) Stage 2: QA→R LoRA finetune (gold answer)
python vcr_ain/train_ain_lora.py \
  --train-jsonl "${OUT_TRAIN}/vcr_qar.sft.jsonl" \
  --valid-jsonl "${OUT_VAL}/vcr_qar.sft.jsonl" \
  --output-dir "/home/ahmedjaheen/data/ain_lora_qar" \
  --model "${AIN_MODEL}" \
  --input-format qwen \
  --max-length "${MAX_LENGTH}" \
  --bf16 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 2e-5 \
  --num-epochs 2
