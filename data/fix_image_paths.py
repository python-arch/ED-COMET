import json
import os

IMAGE_FOLDER = "/content/drive/MyDrive/gen_photos"
INPUT_FILE = "qwen3_train.json"
OUTPUT_FILE = "qwen3_colab_train.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

for entry in data:
    filename = os.path.basename(entry["image"])
    entry["image"] = os.path.join(IMAGE_FOLDER, filename)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Fixed. Output saved to {OUTPUT_FILE}")
