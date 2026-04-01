from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.preprocess import preprocess_image

print("Loading TrOCR model... (first run downloads ~1.3GB, please wait)")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()

print("TrOCR model loaded successfully!")


def split_into_lines(pil_image: Image.Image) -> list:
    """Split a document image into individual line images"""
    # Convert to numpy for OpenCV processing
    img_array = np.array(pil_image.convert("L"))

    # Binarize
    _, binary = cv2.threshold(
        img_array, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Sum pixel values horizontally to find lines
    horizontal_sum = np.sum(binary, axis=1)

    # Find line boundaries — rows with text vs empty rows
    threshold = np.max(horizontal_sum) * 0.05
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

    # Handle case where last line reaches bottom of image
    if in_line:
        line_ends.append(len(horizontal_sum))

    # Crop each line with padding
    lines = []
    padding = 5
    width = pil_image.width

    for start, end in zip(line_starts, line_ends):
        # Skip lines that are too thin (noise)
        if end - start < 8:
            continue
        top = max(0, start - padding)
        bottom = min(pil_image.height, end + padding)
        line_img = pil_image.crop((0, top, width, bottom))
        lines.append(line_img)

    return lines


def ocr_single_line(line_image: Image.Image) -> str:
    """Run TrOCR on a single line image"""
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
    """Full pipeline — preprocess, split lines, OCR each line"""
    # Preprocess image
    image = preprocess_image(image_path)

    # Split into lines
    lines = split_into_lines(image)

    if not lines:
        # Fallback — run on whole image if no lines detected
        return ocr_single_line(image)

    print(f"Detected {len(lines)} lines in document")

    # OCR each line
    extracted_lines = []
    for i, line in enumerate(lines):
        print(f"Processing line {i + 1}/{len(lines)}...")
        text = ocr_single_line(line)
        if text:
            extracted_lines.append(text)

    # Join all lines
    full_text = "\n".join(extracted_lines)
    return full_text


## Save and test again

#The server will auto-reload since we used `--reload`. Just go back to Swagger UI and upload the same image again.

#This time the terminal will show:

#Detected X lines in document
#Processing line 1/X...
#Processing line 2/X...
