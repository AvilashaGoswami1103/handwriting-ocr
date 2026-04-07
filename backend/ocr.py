import easyocr
from PIL import Image
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from backend.preprocess import preprocess_image

# EasyOCR (detect text regions)
reader = easyocr.Reader(['en'], gpu=False)

# TrOCR (refine text)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()


def trocr_refine(crop: Image.Image) -> str:
    crop = crop.resize((384, 384))

    pixel_values = processor(images=crop, return_tensors="pt").pixel_values

    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=64)

    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def run_ocr(image_path: str) -> str:
    image = preprocess_image(image_path)

    img_np = np.array(image)

    print("Running EasyOCR...")

    results = reader.readtext(img_np)

    if not results:
        print("No text detected")
        return ""

    final_text = []

    for (bbox, text, confidence) in results:
        print("Detected:", text, "| Conf:", confidence)

        if confidence < 0.3:
            continue

        # Get bounding box
        x_min = int(min([p[0] for p in bbox]))
        y_min = int(min([p[1] for p in bbox]))
        x_max = int(max([p[0] for p in bbox]))
        y_max = int(max([p[1] for p in bbox]))

        crop = image.crop((x_min, y_min, x_max, y_max))

        try:
            refined = trocr_refine(crop)
            final_text.append(refined)
        except Exception as e:
            print("Refine failed:", e)
            final_text.append(text)

    return "\n".join(final_text)