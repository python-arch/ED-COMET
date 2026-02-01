#!/usr/bin/env bash
set -euo pipefail

# Allow KEY=VALUE args to act as env overrides.
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    export "$kv"
  fi
done

echo "=================================================="
echo "COMET ATOMIC finetune with fairseq (.pt checkpoint)"
echo "=================================================="

# ---- configuration (edit or override via env vars) ----
FAIRSEQ_PREPROCESS=${FAIRSEQ_PREPROCESS:-fairseq-preprocess}
FAIRSEQ_TRAIN=${FAIRSEQ_TRAIN:-fairseq-train}
FAIRSEQ_GENERATE=${FAIRSEQ_GENERATE:-fairseq-generate}
FAIRSEQ_INTERACTIVE=${FAIRSEQ_INTERACTIVE:-fairseq-interactive}
PYTHON_BIN=${PYTHON_BIN:-python}

# Paths
ATOMIC_BART_DATA=${ATOMIC_BART_DATA:-"./data/atomic_bart"}           # has train/val/test.{source,target}
DATA_BIN=${DATA_BIN:-"./data-bin/atomic_bart"}                       # output binarized data
CKPT_PATH=${CKPT_PATH:-"/path/to/checkpoint_best.pt"}                # fairseq .pt checkpoint
DICT_PATH=${DICT_PATH:-"/path/to/dict.txt"}                          # dict used for pretraining
SAVE_DIR=${SAVE_DIR:-"./checkpoints/comet_finetune_fairseq"}         # output checkpoints

# Optional: expand vocab to include COMET tokens
EXPAND_TOKENS=${EXPAND_TOKENS:-1}
EXPAND_FROM_DATA=${EXPAND_FROM_DATA:-0}
MIN_FREQ=${MIN_FREQ:-5}
EXPANDED_CKPT=${EXPANDED_CKPT:-"./checkpoints/checkpoint_best_comet.pt"}
EXPANDED_DICT=${EXPANDED_DICT:-"./data-bin/atomic_bart/dict_comet.txt"}

# Optional: per-epoch eval
EVAL_EACH_EPOCH=${EVAL_EACH_EPOCH:-0}
EVAL_MODE=${EVAL_MODE:-"simple"}  # simple | paper
EVAL_DURING_TRAINING=${EVAL_DURING_TRAINING:-0}
EVAL_SPLIT=${EVAL_SPLIT:-"valid"}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-"./results_fairseq"}
EVAL_BEAM=${EVAL_BEAM:-1}
EVAL_MAX_LEN_B=${EVAL_MAX_LEN_B:-24}
EVAL_LOG_FILE=${EVAL_LOG_FILE:-"$EVAL_OUT_DIR/rouge_log.jsonl"}
SYSTEM_EVAL_DIR=${SYSTEM_EVAL_DIR:-"./system_eval"}

# Train hyperparams
ARCH=${ARCH:-bart_large}
NUM_GPUS=${NUM_GPUS:-1}
MAX_EPOCH=${MAX_EPOCH:-10}
MAX_TOKENS=${MAX_TOKENS:-2048}
UPDATE_FREQ=${UPDATE_FREQ:-4}
LR=${LR:-3e-5}
LR_SCHEDULER=${LR_SCHEDULER:-fixed}
TOTAL_NUM_UPDATE=${TOTAL_NUM_UPDATE:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
NO_EPOCH_CHECKPOINTS=${NO_EPOCH_CHECKPOINTS:-0}
NO_LAST_CHECKPOINTS=${NO_LAST_CHECKPOINTS:-0}
KEEP_BEST_CHECKPOINTS=${KEEP_BEST_CHECKPOINTS:-1}
KEEP_LAST_EPOCHS=${KEEP_LAST_EPOCHS:-1}

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Missing checkpoint: $CKPT_PATH"
  exit 1
fi
if [[ ! -f "$DICT_PATH" ]]; then
  echo "Missing dict: $DICT_PATH"
  exit 1
fi
if [[ ! -f "$ATOMIC_BART_DATA/train.source" ]]; then
  echo "Missing train.source in $ATOMIC_BART_DATA"
  exit 1
fi

DICT_TO_USE="$DICT_PATH"
CKPT_TO_USE="$CKPT_PATH"

if [[ "$EXPAND_TOKENS" -eq 1 || "$EXPAND_FROM_DATA" -eq 1 ]]; then
  echo "Step 0: Expanding vocab..."
  CKPT_PATH="$CKPT_PATH" DICT_PATH="$DICT_PATH" ATOMIC_BART_DATA="$ATOMIC_BART_DATA" \
  EXPAND_TOKENS="$EXPAND_TOKENS" EXPAND_FROM_DATA="$EXPAND_FROM_DATA" MIN_FREQ="$MIN_FREQ" \
  EXPANDED_CKPT="$EXPANDED_CKPT" EXPANDED_DICT="$EXPANDED_DICT" \
  $PYTHON_BIN - <<'PY'
import os
from pathlib import Path
import torch
from collections import Counter

ckpt_path = Path(os.environ["CKPT_PATH"])
dict_path = Path(os.environ["DICT_PATH"])
data_dir = Path(os.environ["ATOMIC_BART_DATA"])
expand_tokens = os.environ.get("EXPAND_TOKENS", "1") == "1"
expand_from_data = os.environ.get("EXPAND_FROM_DATA", "0") == "1"
min_freq = int(os.environ.get("MIN_FREQ", "5"))
out_ckpt = Path(os.environ["EXPANDED_CKPT"])
out_dict = Path(os.environ["EXPANDED_DICT"])

special_tokens = [
    "PersonX", "PersonY", "PersonZ", "___", "[GEN]",
    "AtLocation", "CapableOf", "Causes", "CausesDesire", "CreatedBy",
    "DefinedAs", "DesireOf", "Desires", "HasA", "HasFirstSubevent",
    "HasLastSubevent", "HasPainCharacter", "HasPainIntensity", "HasPrerequisite",
    "HasProperty", "HasSubEvent", "HasSubevent", "HinderedBy", "InheritsFrom",
    "InstanceOf", "IsA", "LocatedNear", "LocationOfAction", "MadeOf", "MadeUpOf",
    "MotivatedByGoal", "NotCapableOf", "NotDesires", "NotHasA", "NotHasProperty",
    "NotIsA", "NotMadeOf", "ObjectUse", "PartOf", "ReceivesAction", "RelatedTo",
    "SymbolOf", "UsedFor", "isAfter", "isBefore", "isFilledBy",
    "oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed",
    "xReact", "xReason", "xWant",
]

def read_dict(path):
    vocab = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            vocab.append(tok)
    return vocab

def load_dict_from_checkpoint(path):
    try:
        from fairseq import checkpoint_utils
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(path)], strict=False)
        return task.source_dictionary
    except Exception as exc:
        print(f"Warning: could not load dictionary from checkpoint: {exc}")
        return None

def iter_data_tokens(data_dir):
    counts = Counter()
    files = [
        data_dir / "train.source",
        data_dir / "train.target",
        data_dir / "val.source",
        data_dir / "val.target",
        data_dir / "test.source",
        data_dir / "test.target",
    ]
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                for tok in line.strip().split():
                    counts[tok] += 1
    return counts

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"] if "model" in ckpt else ckpt
old_vocab_size = None
if "encoder.embed_tokens.weight" in state:
    old_vocab_size = state["encoder.embed_tokens.weight"].shape[0]

d = load_dict_from_checkpoint(ckpt_path)
if d is not None:
    orig_vocab_size = len(d)
    to_add = []
    if expand_tokens:
        to_add.extend([t for t in special_tokens if t not in d.indices])
    if expand_from_data:
        counts = iter_data_tokens(data_dir)
        data_tokens = [t for t, c in counts.items() if c >= min_freq]
        data_tokens = sorted(data_tokens, key=lambda x: (-counts[x], x))
        to_add.extend([t for t in data_tokens if t not in d.indices])
    for t in to_add:
        d.add_symbol(t)
    new_vocab_size = len(d)
    out_dict.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(out_dict))
    added_tokens = to_add
else:
    vocab = read_dict(dict_path)
    vocab_set = set(vocab)
    to_add = []
    if expand_tokens:
        to_add.extend([t for t in special_tokens if t not in vocab_set])
    if expand_from_data:
        counts = iter_data_tokens(data_dir)
        data_tokens = [t for t, c in counts.items() if c >= min_freq]
        data_tokens = sorted(data_tokens, key=lambda x: (-counts[x], x))
        to_add.extend([t for t in data_tokens if t not in vocab_set])
    dict_vocab_size = len(vocab)
    orig_vocab_size = dict_vocab_size
    new_vocab_size = dict_vocab_size + len(to_add)
    out_dict.parent.mkdir(parents=True, exist_ok=True)
    with out_dict.open("w", encoding="utf-8") as f:
        with dict_path.open("r", encoding="utf-8") as src:
            for line in src:
                f.write(line)
        for t in to_add:
            f.write(f"{t} 1\n")
    added_tokens = to_add

def expand_weight(key, new_size):
    if key not in state:
        return
    w = state[key]
    old_size, dim = w.shape
    if old_size == new_size:
        return
    if old_size > new_size:
        raise ValueError(f"{key} size {old_size} > new vocab size {new_size}")
    new_w = w.new_empty(new_size, dim)
    new_w[:old_size] = w
    new_w[old_size:] = 0.02 * torch.randn(new_size - old_size, dim)
    state[key] = new_w

def expand_bias(key, new_size):
    if key not in state:
        return
    b = state[key]
    old_size = b.shape[0]
    if old_size == new_size:
        return
    if old_size > new_size:
        raise ValueError(f"{key} size {old_size} > new vocab size {new_size}")
    new_b = b.new_zeros(new_size)
    new_b[:old_size] = b
    state[key] = new_b

if old_vocab_size is not None and old_vocab_size != orig_vocab_size:
    if old_vocab_size > new_vocab_size:
        raise ValueError(
            f"dict size {orig_vocab_size} < encoder vocab size {old_vocab_size}; "
            "provide a dict with at least the checkpoint vocab size"
        )
    print(
        f"Warning: dict size {orig_vocab_size} != encoder vocab size {old_vocab_size}. "
        f"Expanding to {new_vocab_size} using dict tokens."
    )

expand_weight("encoder.embed_tokens.weight", new_vocab_size)
expand_weight("decoder.embed_tokens.weight", new_vocab_size)
expand_weight("decoder.output_projection.weight", new_vocab_size)
expand_bias("decoder.output_projection.bias", new_vocab_size)

if "model" in ckpt:
    ckpt["model"] = state
else:
    ckpt = {"model": state}

out_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out_ckpt)

print(f"Added tokens: {added_tokens[:50]}")
if len(added_tokens) > 50:
    print(f"... plus {len(added_tokens) - 50} more")
print(f"Expanded vocab: {orig_vocab_size} -> {new_vocab_size}")
print(f"Saved: {out_ckpt}")
print(f"Dict: {out_dict}")
PY
  DICT_TO_USE="$EXPANDED_DICT"
  CKPT_TO_USE="$EXPANDED_CKPT"
fi

echo "Step 1: Binarizing COMET data..."
$FAIRSEQ_PREPROCESS \
  --source-lang source --target-lang target \
  --trainpref "$ATOMIC_BART_DATA/train" \
  --validpref "$ATOMIC_BART_DATA/val" \
  --testpref "$ATOMIC_BART_DATA/test" \
  --destdir "$DATA_BIN" \
  --srcdict "$DICT_TO_USE" --tgtdict "$DICT_TO_USE" \
  --workers 8

echo "Step 2: Finetuning..."
if [[ "$LR_SCHEDULER" == "polynomial_decay" && -z "$TOTAL_NUM_UPDATE" ]]; then
  echo "LR_SCHEDULER=polynomial_decay requires TOTAL_NUM_UPDATE. Set TOTAL_NUM_UPDATE or use LR_SCHEDULER=fixed."
  exit 1
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  if [[ "$FAIRSEQ_TRAIN" == "fairseq-train" ]]; then
    TRAIN_CMD=(torchrun --nproc_per_node "$NUM_GPUS" --module fairseq_cli.train)
  else
    TRAIN_CMD=(torchrun --nproc_per_node "$NUM_GPUS" $FAIRSEQ_TRAIN)
  fi
else
  TRAIN_CMD=($FAIRSEQ_TRAIN)
fi

run_train() {
  local restore_file=$1
  local max_epoch=$2
  local reset_flags=$3

  local ckpt_flags=()
  ckpt_flags+=(--save-interval "$SAVE_INTERVAL")
  ckpt_flags+=(--keep-best-checkpoints "$KEEP_BEST_CHECKPOINTS")
  ckpt_flags+=(--keep-last-epochs "$KEEP_LAST_EPOCHS")
  if [[ "$NO_EPOCH_CHECKPOINTS" -eq 1 ]]; then
    ckpt_flags+=(--no-epoch-checkpoints)
  fi
  if [[ "$NO_LAST_CHECKPOINTS" -eq 1 ]]; then
    ckpt_flags+=(--no-last-checkpoints)
  fi

  "${TRAIN_CMD[@]}" "$DATA_BIN" \
    --restore-file "$restore_file" \
    --arch "$ARCH" \
    --task translation \
    --source-lang source --target-lang target \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-08 \
    --lr "$LR" --lr-scheduler "$LR_SCHEDULER" --warmup-updates 500 \
    ${TOTAL_NUM_UPDATE:+--total-num-update "$TOTAL_NUM_UPDATE"} \
    --max-tokens "$MAX_TOKENS" --update-freq "$UPDATE_FREQ" \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --clip-norm 0.1 \
    --max-epoch "$max_epoch" \
    "${ckpt_flags[@]}" \
    $reset_flags \
    --save-dir "$SAVE_DIR"
}

run_eval() {
  local ckpt=$1
  if [[ "$EVAL_MODE" == "paper" ]]; then
    bash scripts/eval_fairseq_paper_style.sh \
      FAIRSEQ_INTERACTIVE="$FAIRSEQ_INTERACTIVE" \
      PYTHON_BIN="$PYTHON_BIN" \
      DATA_BIN="$DATA_BIN" \
      CKPT="$ckpt" \
      SYSTEM_EVAL_DIR="$SYSTEM_EVAL_DIR" \
      OUT_DIR="$EVAL_OUT_DIR" \
      LOG_FILE="$EVAL_OUT_DIR/paper_eval_log.jsonl" \
      BEAM="$EVAL_BEAM" \
      MAX_LEN_B="$EVAL_MAX_LEN_B"
  else
    bash scripts/eval_fairseq_checkpoint.sh \
      FAIRSEQ_GENERATE="$FAIRSEQ_GENERATE" \
      DATA_BIN="$DATA_BIN" \
      CKPT="$ckpt" \
      SPLIT="$EVAL_SPLIT" \
      OUT_DIR="$EVAL_OUT_DIR" \
      LOG_FILE="$EVAL_LOG_FILE" \
      BEAM="$EVAL_BEAM" \
      MAX_LEN_B="$EVAL_MAX_LEN_B"
  fi
}

if [[ "$EVAL_EACH_EPOCH" -eq 1 && "$EVAL_DURING_TRAINING" -eq 1 ]]; then
  for epoch in $(seq 1 "$MAX_EPOCH"); do
    if [[ "$epoch" -eq 1 ]]; then
      run_train "$CKPT_TO_USE" "$epoch" "--reset-optimizer --reset-dataloader --reset-meters"
    else
      last_ckpt="$SAVE_DIR/checkpoint_last.pt"
      if [[ ! -f "$last_ckpt" ]]; then
        last_ckpt="$CKPT_TO_USE"
      fi
      run_train "$last_ckpt" "$epoch" ""
    fi
    ckpt_path="$SAVE_DIR/checkpoint${epoch}.pt"
    if [[ ! -f "$ckpt_path" ]]; then
      ckpt_path="$SAVE_DIR/checkpoint_last.pt"
    fi
    run_eval "$ckpt_path"
  done
else
  run_train "$CKPT_TO_USE" "$MAX_EPOCH" "--reset-optimizer --reset-dataloader --reset-meters"
fi

echo ""
echo "=================================================="
echo "Finetune complete: $SAVE_DIR"
echo "=================================================="
echo "Optional generation:"
echo "  $FAIRSEQ_GENERATE $DATA_BIN --path $SAVE_DIR/checkpoint_best.pt --source-lang source --target-lang target --beam 5 --max-len-b 24"

if [[ "$EVAL_EACH_EPOCH" -eq 1 && "$EVAL_DURING_TRAINING" -eq 0 ]]; then
  echo ""
  echo "Running per-epoch eval ($EVAL_MODE)..."
  mkdir -p "$EVAL_OUT_DIR"
  while IFS= read -r ckpt; do
    run_eval "$ckpt"
  done < <(ls "$SAVE_DIR"/checkpoint*.pt 2>/dev/null | sort -V)
  if [[ "$EVAL_MODE" == "paper" ]]; then
    echo "Eval log: $EVAL_OUT_DIR/paper_eval_log.jsonl"
  else
    echo "Eval log: $EVAL_LOG_FILE"
  fi
fi
