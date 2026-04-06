# This cleans up the image before TrOCR sees it — fixes skew, removes noise, improves contrast. 

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path: str) -> Image.Image:
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Check if image has colored ink (non-black text)
    # by comparing color channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert to grayscale — use better method for colored ink
    # This inverts the channel weighting to preserve colored ink
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to detect if text is dark or light colored
    # For colored ink (red, blue) use channel subtraction
    b, g, r = cv2.split(img)

    # For red ink — red channel is high, others are low
    # Subtract to get better contrast
    red_ink = cv2.subtract(r, g)
    blue_ink = cv2.subtract(b, g)

    # Pick the channel with most contrast
    contrasts = {
        'gray': gray.std(),
        'red': red_ink.std(),
        'blue': blue_ink.std()
    }
    best_channel = max(contrasts, key=contrasts.get)

    if best_channel == 'red':
        working = red_ink
    elif best_channel == 'blue':
        working = blue_ink
    else:
        working = gray

    # Denoise
    denoised = cv2.fastNlMeansDenoising(working, h=10)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    binary = contrast  # keep grayscale, DO NOT threshold

    # Deskew
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) > 0.5:
            (h, w) = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    pil_image = Image.fromarray(binary).convert("RGB")
    return pil_image