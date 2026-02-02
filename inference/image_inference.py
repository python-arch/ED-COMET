import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_DIR = "/content/qwen2-vl-atomic-finetune"
IMAGE_DIR = "/content/drive/MyDrive/gen_photos"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    offload_buffers=True
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

def test_single_image(image_path):
    conversation = [{
        "from": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Analyze the ATOMIC social relations in this image."}
        ]
    }]

    text_input = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info([conversation])

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300)

    return processor.decode(output_ids[0], skip_special_tokens=True)


for fname in os.listdir(IMAGE_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        print(fname)
        print(test_single_image(os.path.join(IMAGE_DIR, fname)))
