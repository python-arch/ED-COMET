#!/usr/bin/env python3
import argparse
import ast
import json
import os
from typing import Any, List

from datasets import load_dataset
from PIL import Image, ImageDraw


def safe_literal_eval(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return default
    return value


def parse_answer_choices(answer_choices: Any) -> List[str]:
    choices = safe_literal_eval(answer_choices, [])
    if not isinstance(choices, list):
        return []
    parsed: List[str] = []
    for choice in choices:
        if isinstance(choice, list):
            parsed.append(" ".join(str(x) for x in choice))
        else:
            parsed.append(str(choice))
    return parsed


def draw_question_boxes(img: Image.Image, question: Any, boxes: Any) -> Image.Image:
    if not isinstance(img, Image.Image):
        return img
    q_tokens = safe_literal_eval(question, [])
    box_list = safe_literal_eval(boxes, [])
    if not isinstance(q_tokens, list) or not isinstance(box_list, list):
        return img
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for word in q_tokens:
        if isinstance(word, list) and word:
            idx = word[0]
            if isinstance(idx, int) and 0 <= idx < len(box_list):
                box = box_list[idx]
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                label = f"person{idx + 1}"
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1, y1 - 10), label, fill="white")
    return img_copy


def select_image_source(ex: dict) -> Any:
    for key in ("image", "img"):
        if ex.get(key) is not None:
            return ex.get(key)
    if ex.get("img_fn") is not None:
        return ex.get("img_fn")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EG-VCR split to VCR-style JSONL (Q->A).")
    parser.add_argument("--dataset", default="CulTex-VLM/EG-VCR")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--images-dir", default="", help="Optional directory to save images.")
    parser.add_argument("--image-format", default="png", help="Image format when saving.")
    parser.add_argument("--draw-boxes", action="store_true", help="Overlay person boxes.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--id-field",
        default="index",
        help="Field to use as qid (use 'index' for dataset index).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    if args.images_dir:
        os.makedirs(args.images_dir, exist_ok=True)

    letters = ["A", "B", "C", "D"]
    total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(dataset):
            if args.max_examples and idx >= args.max_examples:
                break
            q_text = ex.get("question_orig") or ex.get("question") or ""
            choices = parse_answer_choices(ex.get("answer_choices"))
            if len(choices) != 4:
                continue
            prompt = "Question: " + q_text + "\nOptions: " + " ".join(
                [f"{letters[i]}) {choices[i]}" for i in range(len(choices))]
            )
            answer_label = ex.get("answer_label")
            target = letters[answer_label] if isinstance(answer_label, int) else ""
            if args.id_field == "index":
                qid = idx
            else:
                qid = ex.get(args.id_field, idx)
            image_path = ""
            if args.images_dir:
                img = select_image_source(ex)
                if hasattr(img, "save"):
                    if args.draw_boxes:
                        img = draw_question_boxes(img, ex.get("question"), ex.get("boxes"))
                    filename = f"egvcr_{idx}.{args.image_format}"
                    image_path = os.path.join(args.images_dir, filename)
                    try:
                        if args.image_format.lower() in ("jpg", "jpeg") and hasattr(img, "mode"):
                            if img.mode not in ("RGB", "L"):
                                img = img.convert("RGB")
                        img.save(image_path)
                    except Exception:
                        image_path = ""
                elif isinstance(img, str):
                    image_path = img

            rec = {
                "index": idx,
                "qid": qid,
                "img": image_path,
                "question": q_text,
                "options": {letters[i]: choices[i] for i in range(len(choices))},
                "prompt": prompt,
                "target": target,
                "task": "qa",
            }
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            total += 1

    print(f"Wrote {total} QA records to {args.output}")


if __name__ == "__main__":
    main()
