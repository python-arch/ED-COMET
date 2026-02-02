import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from settings import MODEL_ID, DATASET_PATH, OUTPUT_DIR, IMAGE_DIR
from collator import make_collate_fn
from lora_config import get_lora_config

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    offload_buffers=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

train_dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

collate_fn = make_collate_fn(processor, IMAGE_DIR)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataset_text_field="conversations",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    args=training_args,
    peft_config=get_lora_config(),
)

print("Starting fine-tuning")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print("Model saved")
