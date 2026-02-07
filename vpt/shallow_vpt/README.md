# Visual Prompt Tuning (VPT) for Visual Commonsense Reasoning

This repository contains implementations of **Visual Prompt Tuning (VPT-Shallow)** applied to CLIP models for Visual Commonsense Reasoning (VCR) tasks. The experiments are conducted on two different VCR datasets.

## Notebooks Overview

|  Dataset | Description |
|----------|-------------|
| GD-VCR   | VPT-Shallow on the GD-VCR dataset with 4 regions (East Asia, South Asia, Africa, West) |
| EC-VCR (our work of Egyptian Culture dataset) | VPT-Shallow on the EC-VCR dataset|

---

## What is Visual Prompt Tuning (VPT)?
Visual Prompt Tuning is a parameter-efficient fine-tuning technique that adds learnable prompt tokens to the vision transformer's input sequence while keeping the pre-trained backbone frozen. This approach enables:

- **Parameter Efficiency**: Only ~0.01% of total parameters are trainable
- **Fast Training**: Significantly reduced training time compared to full fine-tuning
- **Preserved Knowledge**: Pre-trained CLIP knowledge remains intact

---
## Dataset Structure

Both notebooks expect data in **JSONL format** with the following schema:

```json
{
    "img_fn": "path/to/image.jpg",
    "objects": ["person", "dog", "car", ...],
    "question": ["What", "is", [0], "doing", "?"],
    "answer_choices": [
        ["Walking", "the", [1], "."],
        ["Driving", "the", [2], "."],
    ],
    "answer_label": 0
}
```
---

## Output Metrics

### Training Metrics
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch

---

## Technical Details

### Model Architecture
- **Base Model**: CLIP ViT-Large/14 (`openai/clip-vit-large-patch14`)
- **Vision Encoder**: ViT-L/14 (24 layers, 1024 hidden dim)
- **Text Encoder**: Transformer (12 layers, 768 hidden dim)
- **Total Parameters**: ~428M
- **Trainable Parameters**: ~51K (prompt tokens only)

### Training Details
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup
- **Loss Function**: Cross-Entropy Loss

### Prompt Injection Strategy
Prompts are injected after the [CLS] token:
```
Sequence: [CLS] [P1] [P2] ... [Pn] [Patch1] [Patch2] ... [PatchM]
```

---