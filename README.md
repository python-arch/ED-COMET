# Qwen2-VL EG-Atomic Fine-Tuning and Evaluation

This repository contains a project for fine-tuning **Qwen2-VL** on multimodal ATOMIC-style datasets and performing **slot-filling evaluation** on events and images. It supports both **image-based inference** and **text-field masking evaluation**.

---

## Project Structure

```
qwen2_vl_atomic/
│
├── data/
│   ├── fix_image_paths.py      # Script to fix JSON image paths for Colab
│   ├── qwen3_train.json        # Original dataset
│   └── qwen3_colab_train.json  # Fixed dataset with correct paths for google colab usage
│
├── training/
│   ├── train.py                # Main fine-tuning script
│   ├── collator.py             # Custom data collator for multimodal data
│   ├── lora_config.py          # LoRA configuration
│   └── settings.py             # Centralized paths and hyperparameters
│
├── inference/
│   ├── image_inference.py      # Test the model on individual images
│   └── slot_filling.py         # Slot-filling for masked text fields
│
├── evaluation/
│   └── evaluate_atomic.py      # Evaluate model predictions vs ground truth
│
├── requirements.txt
└── README.md
```

---

## Features

* Fine-tune **Qwen2-VL-2B-Instruct** with **LoRA** for multimodal understanding.
* Automatically injects **image paths** into JSON datasets for Colab compatibility.
* Supports **image-based ATOMIC social relation inference**.
* Slot-filling evaluation for masked event fields.
* Computes **Exact Match** and **Overlap (%)** metrics for predicted fields.

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

> Note: Make sure you have a GPU runtime for fine-tuning.

---

## Data Preparation

1. Place your images in `data/gen_photos/`.
2. Update your dataset paths inside `data/fix_image_paths.py`.
3. Run:

```bash
python data/fix_image_paths.py
```

This will generate `qwen3_colab_train.json` with updated image paths.

---

## Fine-Tuning

Run the training script:

```bash
python training/train.py
```

Key points:

* Uses **4-bit quantization** for memory efficiency.
* Fine-tunes with **LoRA** on selected modules.
* Output saved in `training.OUTPUT_DIR`.

---

## Image-Based Inference

Test the fine-tuned model on images:

```bash
python inference/image_inference.py
```

* Generates ATOMIC-style social relations per image.
* Works with `.png`, `.jpg`, and `.jpeg` images in the specified folder.

---

## Slot-Filling Evaluation

Evaluate masked fields in events:

```bash
python evaluation/evaluate_atomic.py
```

* Fields include: `oEffect`, `oReact`, `oWant`, `xAttr`, `xEffect`, `xIntent`, `xNeed`, `xReact`, `xWant`.
* Outputs **Exact Match** and **Overlap (%)** for each field.

---

## Configuration

Modify paths, model IDs, or hyperparameters in:

* `training/settings.py` – paths and training options.
* `training/lora_config.py` – LoRA hyperparameters.

---

### Test Results

Example evaluation of masked fields using the fine-tuned model:

| Field   | GroundTruth                                | Predicted                     | Overlap(%) |
| ------- | ------------------------------------------ | ----------------------------- | ---------- |
| xNeed   | ['cleaning supplies', 'organize schedule'] | ['cleaning supplies']         | 75.0       |
| xIntent | ['to prepare mosque for Eid worship']      | ['to prepare mosque for Eid'] | 80.0       |
| xReact  | ['satisfied', 'humble']                    | ['satisfied', 'modest']       | 60.0       |

---

## Notes

* Ensure `torch` and `transformers` versions are compatible with your GPU.
* Disable **W&B logging** via `settings.py` for Colab.
* Image-based and text-based evaluations can run independently.

---

## References

* [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [Transformers Documentation](https://huggingface.co/docs/transformers)
