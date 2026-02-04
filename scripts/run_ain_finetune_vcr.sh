#!/usr/bin/env bash
set -euo pipefail

VCR_ROOT="/home/ahmedjaheen/data/vcr1"
OUT_TRAIN="/home/ahmedjaheen/data/vcr1_out"
OUT_VAL="/home/ahmedjaheen/data/vcr1_out_val"
AIN_MODEL="MBZUAI/AIN"
MAX_LENGTH=4096
IMAGE_MAX_PIXELS=589824
BATCH_SIZE=4
GRAD_ACC=4
NUM_EPOCHS=1
SUBSET_TRAIN=30000
SUBSET_VAL=3000

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

SFT_TRAIN_QA="${OUT_TRAIN}/vcr_qa.sft.jsonl"
SFT_VAL_QA="${OUT_VAL}/vcr_qa.sft.jsonl"
if [[ "${SUBSET_TRAIN}" -gt 0 ]]; then
  SFT_TRAIN_QA="${OUT_TRAIN}/vcr_qa.sft.${SUBSET_TRAIN}.jsonl"
  SFT_VAL_QA="${OUT_VAL}/vcr_qa.sft.${SUBSET_VAL}.jsonl"
  head -n "${SUBSET_TRAIN}" "${OUT_TRAIN}/vcr_qa.sft.jsonl" > "${SFT_TRAIN_QA}"
  head -n "${SUBSET_VAL}" "${OUT_VAL}/vcr_qa.sft.jsonl" > "${SFT_VAL_QA}"
fi

# 3) Stage 1: Q→A LoRA finetune
python vcr_ain/train_ain_lora.py \
  --train-jsonl "${SFT_TRAIN_QA}" \
  --valid-jsonl "${SFT_VAL_QA}" \
  --output-dir "/home/ahmedjaheen/data/ain_lora_qa" \
  --model "${AIN_MODEL}" \
  --input-format qwen \
  --max-length "${MAX_LENGTH}" \
  --image-max-pixels "${IMAGE_MAX_PIXELS}" \
  --bf16 \
  --per-device-batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACC}" \
  --lr 2e-5 \
  --num-epochs "${NUM_EPOCHS}"

# 4) Build SFT JSONL for QA→R
python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_TRAIN}/vcr_qar.jsonl" \
  --output "${OUT_TRAIN}/vcr_qar.sft.jsonl" \
  --format qwen

python vcr_ain/prepare_ain_sft.py \
  --input "${OUT_VAL}/vcr_qar.jsonl" \
  --output "${OUT_VAL}/vcr_qar.sft.jsonl" \
  --format qwen

SFT_TRAIN_QAR="${OUT_TRAIN}/vcr_qar.sft.jsonl"
SFT_VAL_QAR="${OUT_VAL}/vcr_qar.sft.jsonl"
if [[ "${SUBSET_TRAIN}" -gt 0 ]]; then
  SFT_TRAIN_QAR="${OUT_TRAIN}/vcr_qar.sft.${SUBSET_TRAIN}.jsonl"
  SFT_VAL_QAR="${OUT_VAL}/vcr_qar.sft.${SUBSET_VAL}.jsonl"
  head -n "${SUBSET_TRAIN}" "${OUT_TRAIN}/vcr_qar.sft.jsonl" > "${SFT_TRAIN_QAR}"
  head -n "${SUBSET_VAL}" "${OUT_VAL}/vcr_qar.sft.jsonl" > "${SFT_VAL_QAR}"
fi

# 5) Stage 2: QA→R LoRA finetune (gold answer)
python vcr_ain/train_ain_lora.py \
  --train-jsonl "${SFT_TRAIN_QAR}" \
  --valid-jsonl "${SFT_VAL_QAR}" \
  --output-dir "/home/ahmedjaheen/data/ain_lora_qar" \
  --model "${AIN_MODEL}" \
  --input-format qwen \
  --max-length "${MAX_LENGTH}" \
  --image-max-pixels "${IMAGE_MAX_PIXELS}" \
  --bf16 \
  --per-device-batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACC}" \
  --lr 2e-5 \
  --num-epochs "${NUM_EPOCHS}"
