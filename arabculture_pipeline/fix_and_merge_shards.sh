#!/usr/bin/env bash
set -euo pipefail

# Apply tail repairs to two shards, rebuild atomic_bart, then merge into one dataset.
# Edit the paths below or override via environment variables.

SHARD1_DIR=${SHARD1_DIR:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/arabculture_atomic_shard1_qwen14b"}
SHARD2_DIR=${SHARD2_DIR:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/arabculture_atomic_shard2_qwen14b"}
COMBINED_DIR=${COMBINED_DIR:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/arabculture_atomic_combined"}

MODEL=${MODEL:-"Qwen/Qwen2.5-14B-Instruct"}
TP_SIZE=${TP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-128}

REPAIR_SCRIPT=${REPAIR_SCRIPT:-"arabculture_pipeline/repair_tails.py"}
REPAIR_HEADS_SCRIPT=${REPAIR_HEADS_SCRIPT:-"arabculture_pipeline/repair_heads.py"}
BUILD_SCRIPT=${BUILD_SCRIPT:-"arabculture_pipeline/build_atomic_bart_from_jsonl.py"}
HEAD_MODEL=${HEAD_MODEL:-"$MODEL"}
HEAD_BATCH_SIZE=${HEAD_BATCH_SIZE:-64}
HEAD_MAX_WORDS=${HEAD_MAX_WORDS:-16}
DROP_BAD_HEADS=${DROP_BAD_HEADS:-1}

repair_split() {
  local shard_dir=$1
  local split=$2
  local in_file="${shard_dir}/atomic_full.${split}.jsonl"
  local head_out="${shard_dir}/atomic_full.${split}.head_repaired.jsonl"
  local out_file="${shard_dir}/atomic_full.${split}.repaired.jsonl"

  if [[ ! -f "$in_file" ]]; then
    echo "Missing input: $in_file"
    exit 1
  fi

  local drop_flag=""
  if [[ "$DROP_BAD_HEADS" -eq 1 ]]; then
    drop_flag="--drop-on-fail"
  fi

  python "$REPAIR_HEADS_SCRIPT" \
    --input "$in_file" \
    --output "$head_out" \
    --model "$HEAD_MODEL" \
    --tp-size "$TP_SIZE" \
    --batch-size "$HEAD_BATCH_SIZE" \
    --max-head-words "$HEAD_MAX_WORDS" \
    $drop_flag

  python "$REPAIR_SCRIPT" \
    --input "$head_out" \
    --output "$out_file" \
    --model "$MODEL" \
    --tp-size "$TP_SIZE" \
    --batch-size "$BATCH_SIZE"
}

rebuild_shard() {
  local shard_dir=$1
  python "$BUILD_SCRIPT" \
    --input-dir "$shard_dir" \
    --use-repaired
}

merge_splits() {
  local split=$1
  local out_file="${COMBINED_DIR}/atomic_full.${split}.repaired.jsonl"
  cat \
    "${SHARD1_DIR}/atomic_full.${split}.repaired.jsonl" \
    "${SHARD2_DIR}/atomic_full.${split}.repaired.jsonl" \
    > "$out_file"
}

for split in train val test; do
  repair_split "$SHARD1_DIR" "$split"
  repair_split "$SHARD2_DIR" "$split"
done

rebuild_shard "$SHARD1_DIR"
rebuild_shard "$SHARD2_DIR"

mkdir -p "$COMBINED_DIR"
for split in train val test; do
  merge_splits "$split"
done

python "$BUILD_SCRIPT" \
  --input-dir "$COMBINED_DIR" \
  --use-repaired

echo "Done."
echo "Shard1 atomic_bart: ${SHARD1_DIR}/atomic_bart"
echo "Shard2 atomic_bart: ${SHARD2_DIR}/atomic_bart"
echo "Combined atomic_bart: ${COMBINED_DIR}/atomic_bart"
