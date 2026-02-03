#!/usr/bin/env python3
import argparse
import inspect
import json
from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments
from qwen_vl_utils import process_vision_info


class VCRJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def _extract_image_path(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        for part in m.get("content", []):
            if part.get("type") == "image":
                return part.get("image")
    return ""


def _build_texts(processor, rec: Dict[str, Any], input_format: str) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    if input_format == "qwen":
        messages = rec["messages"]
        messages_no_assistant = [m for m in messages if m.get("role") != "assistant"]
        prompt_text = processor.apply_chat_template(
            messages_no_assistant, tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_path = _extract_image_path(messages_no_assistant)
        return prompt_text, full_text, image_path, messages_no_assistant

    prompt = rec["prompt"]
    target = rec["target"]
    prompt_text = prompt + "\nAnswer:"
    full_text = prompt_text + " " + target
    image_path = rec["image"]
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}
    ]
    return prompt_text, full_text, image_path, messages


class VCRCollator:
    def __init__(self, processor, input_format: str, max_length: int):
        self.processor = processor
        self.input_format = input_format
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        images: List[Any] = []
        videos: List[Any] = []

        for rec in batch:
            prompt_text, full_text, _image_path, messages_no_assistant = _build_texts(
                self.processor, rec, self.input_format
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)

            image_inputs, video_inputs = process_vision_info(messages_no_assistant)
            if image_inputs:
                images.append(image_inputs if len(image_inputs) > 1 else image_inputs[0])
            else:
                images.append(None)
            if video_inputs:
                videos.append(video_inputs if len(video_inputs) > 1 else video_inputs[0])
            else:
                videos.append(None)

        has_images = any(img is not None for img in images)
        has_videos = any(vid is not None for vid in videos)

        full_kwargs: Dict[str, Any] = {
            "text": full_texts,
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
        }
        prompt_kwargs: Dict[str, Any] = {
            "text": prompt_texts,
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
        }
        if has_images:
            full_kwargs["images"] = images
            prompt_kwargs["images"] = images
        if has_videos:
            full_kwargs["videos"] = videos
            prompt_kwargs["videos"] = videos

        full_inputs = self.processor(**full_kwargs)
        prompt_inputs = self.processor(**prompt_kwargs)

        if self.max_length and "image_grid_thw" in full_inputs:
            image_tokens = full_inputs["image_grid_thw"].prod(dim=1).to(torch.long)
            max_image_tokens = int(image_tokens.max().item())
            if max_image_tokens > self.max_length:
                raise ValueError(
                    f"max_length ({self.max_length}) is smaller than image tokens ({max_image_tokens}). "
                    "Increase --max-length or reduce image resolution."
                )

        if self.max_length and full_inputs["input_ids"].size(1) > self.max_length:
            seq_len = full_inputs["input_ids"].size(1)
            for key, value in list(full_inputs.items()):
                if isinstance(value, torch.Tensor) and value.dim() == 2 and value.size(1) == seq_len:
                    full_inputs[key] = value[:, : self.max_length]

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        labels = input_ids.clone()

        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).to(torch.long)
        if self.max_length:
            prompt_lens = torch.clamp(prompt_lens, max=input_ids.size(1))
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, : min(plen, labels.size(1))] = -100

        batch_out: Dict[str, torch.Tensor] = dict(full_inputs)
        batch_out["labels"] = labels
        return batch_out


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA finetune AIN on VCR (Q->A or QA->R).")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="MBZUAI/AIN")
    parser.add_argument("--input-format", choices=["qwen", "simple"], default="qwen")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=None,
        help="Optional lower bound on resized image pixel count.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=None,
        help="Optional upper bound on resized image pixel count.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules.",
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        if args.image_min_pixels is not None and hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = args.image_min_pixels
        if args.image_max_pixels is not None and hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = args.image_max_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = VCRJsonlDataset(args.train_jsonl)
    valid_ds = VCRJsonlDataset(args.valid_jsonl)

    collator = VCRCollator(processor, args.input_format, args.max_length)

    train_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to=[],
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        train_kwargs["evaluation_strategy"] = "steps"
    else:
        train_kwargs["eval_strategy"] = "steps"

    train_args = TrainingArguments(**train_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
