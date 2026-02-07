# Visual Prompt Tuning Deep (VPT-Deep) for Visual Commonsense Reasoning

This repository contains implementations of **Visual Prompt Tuning Deep (VPT-Deep)** applied to CLIP models for Visual Commonsense Reasoning (VCR) tasks. Unlike VPT-Shallow which injects prompts only at the input layer, VPT-Deep injects learnable prompts at **every transformer layer**.

## Notebooks Overview

| Dataset | Description |
|---------|-------------|
| GD-VCR | VPT-Deep on the GD-VCR dataset with 4 regions (East Asia, South Asia, Africa, West) |
| EC-VCR (our work of Egyptian Culture dataset) | VPT-Deep on the EC-VCR dataset |

---

## What is VPT-Deep?

Visual Prompt Tuning Deep (VPT-Deep) is an advanced parameter-efficient fine-tuning technique that extends VPT-Shallow by introducing **layer-specific learnable prompts** at every transformer layer. This approach enables:

- **Greater Expressiveness**: Separate prompts per layer allow for layer-specific adaptations
- **Better Task Adaptation**: Deep prompts can influence representations at all depths
- **Parameter Efficiency**: Only small portion of the total parameters are trainable
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
- **Trainable Parameters**: ~1.2M (prompt tokens across all 24 layers)

### Training Details
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup
- **Loss Function**: Cross-Entropy Loss

### Prompt Injection Strategy
Prompts are injected at every transformer layer, replacing the previous layer's prompts:
```
Layer 0: [CLS] [P1_0] [P2_0] ... [Pn_0] [Patch1] [Patch2] ... [PatchM]
Layer 1: [CLS] [P1_1] [P2_1] ... [Pn_1] [Patch1] [Patch2] ... [PatchM]
  ...
Layer N: [CLS] [P1_N] [P2_N] ... [Pn_N] [Patch1] [Patch2] ... [PatchM]
```

---
