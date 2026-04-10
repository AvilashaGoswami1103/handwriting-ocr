# utils.py - Common utilities for the project

import cv2
import numpy as np
from PIL import Image
import os

def preprocess_handwriting(image_path, show_steps=False):
    """Preprocess messy handwriting for better OCR"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    final = cv2.erode(dilated, kernel, iterations=1)
    
    final_pil = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_GRAY2RGB))
    
    if show_steps:
        # Show steps code here (same as before)
        pass
    
    return final_pil, final