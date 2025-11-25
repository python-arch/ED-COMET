# BART Finetuning on ATOMIC2020 - Complete Guide

This guide walks through finetuning pretrained BART model (from fairseq) on the ATOMIC2020 dataset using COMET's training methodology.

## Prerequisites

- BART checkpoint pretrained with fairseq (denoising objective)
- ATOMIC2020 dataset (download from [Google Drive](https://drive.google.com/file/d/1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7/view?usp=drive_link))
- Python 3.x with PyTorch and Transformers

## Step-by-Step Process

### Step 1: Convert Fairseq BART to HuggingFace Format

Since the finetuning script uses HuggingFace Transformers, we need to convert our fairseq checkpoint first:

```bash
# Install transformers if not already installed
pip install transformers torch

# Convert fairseq checkpoint to HuggingFace format
python -m transformers.models.bart.convert_bart_original_pytorch_checkpoint_to_pytorch \
    --fairseq_path /path/to/your/fairseq/checkpoint.pt \
    --pytorch_dump_folder_path ./checkpoints/bart_hf
```

**Note:** Replace `/path/to/your/fairseq/checkpoint.pt` with the actual path to our pretrained checkpoint.

### Step 2: Extract ATOMIC2020 Dataset

Extract the downloaded ATOMIC2020 zip file:

```bash
# Extract the dataset
unzip atomic2020_data-feb2021.zip -d ./atomic2020_data-feb2021

# Check the contents
ls -la ./atomic2020_data-feb2021/
# Should contain: train.tsv, dev.tsv, test.tsv, README.md, LICENSE, etc.
```

The TSV files have the format:
- Column 1: `head` (head event, e.g., "PersonX goes to the store")
- Column 2: `relation` (e.g., "xIntent", "xNeed", "oEffect")
- Column 3: `tail` (tail event, e.g., "to buy groceries")

### Step 3: Convert ATOMIC TSV to BART Training Format

Convert the TSV files to `.source` and `.target` format required by the training script:

```bash
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart
```

This creates:
- `train.source` / `train.target` (training set)
- `val.source` / `val.target` (validation set, from dev.tsv)
- `test.source` / `test.target` (test set)

**Format:**
- `.source` files: `"head_event relation"` (e.g., "PersonX goes to the store xIntent")
- `.target` files: `"tail_event"` (e.g., "to buy groceries")

**Optional:** Add `--add_gen_token` flag to append `[GEN]` to source (for demo model compatibility):
```bash
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart \
    --add_gen_token
```

### Step 4: Finetune BART on ATOMIC2020

Run the finetuning script:

```bash
CUDA_VISIBLE_DEVICES=0 python models/comet_atomic2020_bart/finetune.py \
    --task summarization \
    --num_workers 2 \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 100 \
    --val_check_interval 1.0 \
    --sortish_sampler \
    --max_source_length=48 \
    --max_target_length=24 \
    --val_max_target_length=24 \
    --test_max_target_length=24 \
    --data_dir ./data/atomic_bart \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --output_dir ./results/bart_atomic_finetune \
    --num_train_epochs=10 \
    --model_name_or_path ./checkpoints/bart_hf \
    --atomic
```

**Important Parameters:**
- `--model_name_or_path`: Path to your converted HuggingFace BART checkpoint
- `--data_dir`: Directory containing `.source` and `.target` files
- `--atomic`: **CRITICAL** - Adds ATOMIC relation tokens to the tokenizer
- `--num_train_epochs`: Number of training epochs (adjust as needed)
- `--train_batch_size`: Adjust based on your GPU memory

### Step 5: Use the Complete Workflow Script

Alternatively, you can use the all-in-one script (edit paths first):

```bash
# Edit the script to set your paths
nano scripts/finetune_bart_on_atomic.sh

# Run the complete workflow
bash scripts/finetune_bart_on_atomic.sh
```

## Key Features of COMET Training

The `--atomic` flag (in finetune.py:335-393) automatically:
1. Adds 49 ATOMIC-specific relation tokens to the tokenizer:
   - Social: xIntent, xNeed, xWant, xEffect, xReact, xAttr, etc.
   - Physical: oEffect, oReact, oWant
   - Event: HasSubevent, isAfter, isBefore, etc.
   - ConceptNet: AtLocation, CapableOf, Causes, HasProperty, etc.
2. Resizes the model's token embeddings to accommodate these new tokens
3. Preserves pretrained weights while adding capacity for new tokens

## Training Output

Results will be saved in `./results/bart_atomic_finetune/`:
- Model checkpoints (`*.ckpt`)
- Metrics (`metrics.json`)
- Hyperparameters (`hparams.pkl`)
- Generated predictions

## Testing Your Finetuned Model

After training, test your model:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load your finetuned model
model_path = "./results/bart_atomic_finetune"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate inference
query = "PersonX goes to the store xIntent"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5, num_return_sequences=5)
predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(f"Query: {query}")
print(f"Predictions: {predictions}")
```

Or use the example script:
```bash
# Edit generation_example.py to point to your model
python models/comet_atomic2020_bart/generation_example.py
```

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce `--train_batch_size` and `--eval_batch_size`
- Reduce `--max_source_length` and `--max_target_length`
- Use gradient accumulation: `--accumulate_grad_batches=2`

### Fairseq Conversion Issues
- Ensure your fairseq checkpoint is compatible with BART architecture
- Check transformers library version: `pip install transformers>=4.1.1`

### Data Format Issues
- Verify TSV files have correct delimiter (tab)
- Check for empty lines or malformed entries
- Ensure column order: head, relation, tail

## References

- Paper: [COMET-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs](https://www.semanticscholar.org/paper/COMET-ATOMIC-2020%3A-On-Symbolic-and-Neural-Knowledge-Hwang-Bhagavatula/e39503e01ebb108c6773948a24ca798cd444eb62)
- ATOMIC 2020 Dataset: [Google Drive](https://drive.google.com/file/d/1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7/view?usp=drive_link)
- Pretrained COMET-BART: [Google Storage](https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip)

## Quick Start Commands

```bash
# 1. Convert fairseq checkpoint
python -m transformers.models.bart.convert_bart_original_pytorch_checkpoint_to_pytorch \
    --fairseq_path <your_checkpoint.pt> \
    --pytorch_dump_folder_path ./checkpoints/bart_hf

# 2. Prepare ATOMIC data
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart

# 3. Finetune BART
bash scripts/finetune_bart_on_atomic.sh
```
