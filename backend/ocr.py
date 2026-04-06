from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.preprocess import preprocess_image
from backend.cleanup import cleanup_text

print("Loading TrOCR model... (first run downloads ~1.3GB, please wait)")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.eval()

print("TrOCR model loaded successfully!")


def split_into_lines(pil_image: Image.Image) -> list:
    img_array = np.array(pil_image.convert("L"))

    _, binary = cv2.threshold(
        img_array, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    horizontal_sum = np.sum(binary, axis=1)
    threshold = np.max(horizontal_sum) * 0.15
    in_line = False
    line_starts = []
    line_ends = []

    for i, val in enumerate(horizontal_sum):
        if val > threshold and not in_line:
            in_line = True
            line_starts.append(i)
        elif val <= threshold and in_line:
            in_line = False
            line_ends.append(i)

    if in_line:
        line_ends.append(len(horizontal_sum))

    lines = []
    padding = 5
    width = pil_image.width

    for start, end in zip(line_starts, line_ends):
        if end - start < 15:
            continue
        top = max(0, start - padding)
        bottom = min(pil_image.height, end + padding)
        line_img = pil_image.crop((0, top, width, bottom))
        lines.append(line_img)

    return lines


def ocr_single_line(line_image: Image.Image) -> str:
    line_image = line_image.resize((384, 384))
    pixel_values = processor(
        images=line_image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


def run_ocr(image_path: str) -> str:
    # Step 1 — preprocess
    image = preprocess_image(image_path)

    # Step 2 — split into lines
    lines = split_into_lines(image)

    extracted_lines = []

    if not lines:
        text = ocr_single_line(image)
        if text and len(text.strip()) > 2:
            extracted_lines.append(text)
    else:
        print(f"Detected {len(lines)} lines in document")

        for i, line in enumerate(lines):
            print(f"Processing line {i + 1}/{len(lines)}...")
            text = ocr_single_line(line)

            # 🔥 filtering
            if text and len(text.strip()) > 2:
                extracted_lines.append(text)

    # 🔥 ALWAYS define raw_text
    if extracted_lines:
        raw_text = "\n".join(extracted_lines)
    else:
        raw_text = ""   # fallback to avoid crash

    print("Raw OCR output:")
    print(raw_text)
    print("Running LLM cleanup...")

    # Step 3 — LLM cleanup
    cleaned_text = cleanup_text(raw_text)

    print("Cleaned output:")
    print(cleaned_text)

    return cleaned_text