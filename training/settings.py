import os

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

DATASET_PATH = "qwen_train.json"
IMAGE_DIR = "gen_photos"
OUTPUT_DIR = "qwen2-vl-atomic-finetune"

os.environ["WANDB_DISABLED"] = "true"
