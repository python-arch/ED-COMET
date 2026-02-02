import torch
import re
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_DIR = "/content/qwen2-vl-atomic-finetune"
device = "cuda" if torch.cuda.is_available() else "cpu"

original_text = """Event: PersonX prepares a mosque cleaning volunteer day before Eid prayer.
- oEffect: ['mosque looks tidy', 'worshippers feel comfortable']
- oReact: ['respectful', 'appreciative']
- oWant: ['to pray calmly', 'to invite more volunteers']
- xAttr: ['community-minded', 'devout']
- xEffect: ['feels useful', 'builds rapport']
- xIntent: ['to prepare mosque for Eid worship']
- xNeed: ['cleaning supplies', 'organize schedule']
- xReact: ['satisfied', 'humble']
- xWant: ['to institutionalize event', 'to involve youth']"""

fields = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    offload_buffers=True
)

processor = AutoProcessor.from_pretrained(MODEL_DIR)

def fill_mask(masked_text):
    conversation = [{"from": "user", "content": [{"type": "text", "text": masked_text}]}]
    text_input = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text_input], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True)

def extract_list(text):
    match = re.search(r"(\[[^\]]*\])", text)
    if match:
        try:
            return eval(match.group(1))
        except:
            pass
    return []

results = []

for field in fields:
    masked = re.sub(
        rf"({field}:\s*)\[[^\]]*\]",
        rf"\1[MASK]",
        original_text
    )
    masked += f"\n\nFill the [MASK] for {field} as a Python list."

    pred_raw = fill_mask(masked)
    pred_list = extract_list(pred_raw)
    gt_list = extract_list(original_text.split(field + ":")[1])

    exact = int(pred_list == gt_list)
    overlap = len(set(pred_list) & set(gt_list)) / len(gt_list) * 100 if gt_list else 0

    results.append({
        "Field": field,
        "ExactMatch": exact,
        "Overlap(%)": round(overlap, 2),
        "Predicted": pred_list,
        "GroundTruth": gt_list
    })

df = pd.DataFrame(results)
print(df)
