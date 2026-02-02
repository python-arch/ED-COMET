import os
from qwen_vl_utils import process_vision_info

def make_collate_fn(processor, image_dir):

    def collate_fn(examples):
        conversations = []

        for ex in examples:
            conv = []
            image_path = os.path.join(image_dir, f"{ex['id']}.png")

            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)

            for msg in ex["conversations"]:
                new_msg = {"from": msg["from"], "content": []}

                for c in msg["content"]:
                    if c["type"] == "image":
                        new_msg["content"].append({
                            "type": "image",
                            "image": image_path
                        })
                    else:
                        new_msg["content"].append(c)

                conv.append(new_msg)

            conversations.append(conv)

        texts = [
            processor.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        image_inputs, video_inputs = process_vision_info(conversations)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    return collate_fn
