#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "COMET ATOMIC finetune with fairseq (.pt checkpoint)"
echo "=================================================="

# ---- configuration (edit or override via env vars) ----
FAIRSEQ_PREPROCESS=${FAIRSEQ_PREPROCESS:-fairseq-preprocess}
FAIRSEQ_TRAIN=${FAIRSEQ_TRAIN:-fairseq-train}
FAIRSEQ_GENERATE=${FAIRSEQ_GENERATE:-fairseq-generate}
PYTHON_BIN=${PYTHON_BIN:-python}

# Paths
ATOMIC_BART_DATA=${ATOMIC_BART_DATA:-"./data/atomic_bart"}           # has train/val/test.{source,target}
DATA_BIN=${DATA_BIN:-"./data-bin/atomic_bart"}                       # output binarized data
CKPT_PATH=${CKPT_PATH:-"/path/to/checkpoint_best.pt"}                # fairseq .pt checkpoint
DICT_PATH=${DICT_PATH:-"/path/to/dict.txt"}                          # dict used for pretraining
SAVE_DIR=${SAVE_DIR:-"./checkpoints/comet_finetune_fairseq"}         # output checkpoints

# Optional: expand vocab to include COMET tokens
EXPAND_TOKENS=${EXPAND_TOKENS:-1}
EXPANDED_CKPT=${EXPANDED_CKPT:-"./checkpoints/checkpoint_best_comet.pt"}
EXPANDED_DICT=${EXPANDED_DICT:-"./data-bin/atomic_bart/dict_comet.txt"}

# Train hyperparams
ARCH=${ARCH:-bart_large}
NUM_GPUS=${NUM_GPUS:-1}
MAX_EPOCH=${MAX_EPOCH:-10}
MAX_TOKENS=${MAX_TOKENS:-2048}
UPDATE_FREQ=${UPDATE_FREQ:-4}
LR=${LR:-3e-5}

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

if [[ "$EXPAND_TOKENS" -eq 1 ]]; then
  echo "Step 0: Expanding vocab with COMET tokens..."
  CKPT_PATH="$CKPT_PATH" DICT_PATH="$DICT_PATH" \
  EXPANDED_CKPT="$EXPANDED_CKPT" EXPANDED_DICT="$EXPANDED_DICT" \
  $PYTHON_BIN - <<'PY'
import os
from pathlib import Path
import torch

ckpt_path = Path(os.environ["CKPT_PATH"])
dict_path = Path(os.environ["DICT_PATH"])
out_ckpt = Path(os.environ["EXPANDED_CKPT"])
out_dict = Path(os.environ["EXPANDED_DICT"])

tokens = [
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

vocab = read_dict(dict_path)
vocab_set = set(vocab)
missing = [t for t in tokens if t not in vocab_set]

out_dict.parent.mkdir(parents=True, exist_ok=True)
with out_dict.open("w", encoding="utf-8") as f:
    with dict_path.open("r", encoding="utf-8") as src:
        for line in src:
            f.write(line)
    for t in missing:
        f.write(f"{t} 1\n")

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"] if "model" in ckpt else ckpt

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

orig_vocab_size = len(vocab)
new_vocab_size = orig_vocab_size + len(missing)

if "encoder.embed_tokens.weight" in state:
    old = state["encoder.embed_tokens.weight"].shape[0]
    if old != orig_vocab_size:
        raise ValueError(f"dict size {orig_vocab_size} != encoder vocab size {old}")

expand_weight("encoder.embed_tokens.weight", new_vocab_size)
expand_weight("decoder.embed_tokens.weight", new_vocab_size)
expand_weight("decoder.output_projection.weight", new_vocab_size)

if "model" in ckpt:
    ckpt["model"] = state
else:
    ckpt = state

out_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out_ckpt)

print(f"Added tokens: {missing}")
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
LAUNCH=""
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCH="torchrun --nproc_per_node $NUM_GPUS"
fi

$LAUNCH $FAIRSEQ_TRAIN "$DATA_BIN" \
  --restore-file "$CKPT_TO_USE" \
  --arch "$ARCH" \
  --task translation \
  --source-lang source --target-lang target \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-08 \
  --lr "$LR" --lr-scheduler polynomial_decay --warmup-updates 500 \
  --max-tokens "$MAX_TOKENS" --update-freq "$UPDATE_FREQ" \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.01 --clip-norm 0.1 \
  --max-epoch "$MAX_EPOCH" \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir "$SAVE_DIR"

echo ""
echo "=================================================="
echo "Finetune complete: $SAVE_DIR"
echo "=================================================="
echo "Optional generation:"
echo "  $FAIRSEQ_GENERATE $DATA_BIN --path $SAVE_DIR/checkpoint_best.pt --source-lang source --target-lang target --beam 5 --max-len-b 24"
