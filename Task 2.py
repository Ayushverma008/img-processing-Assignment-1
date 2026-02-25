import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==================== TASK 2: IMAGE ACQUISITION ====================
def load_and_preprocess(image_path, image_id=1):
    """
    Load document image, resize to 512x512, convert to grayscale
    """
    print(f"\n{'='*50}")
    print(f"[DOCUMENT {image_id}]: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return None, None, None
    
    # Get original dimensions
    h, w = img.shape[:2]
    print(f"[INFO] Original dimensions: {w} x {h} pixels")
    
    # Resize to 512x512
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print(f"[OK] Resized to: 512 x 512 pixels")
    print(f"[OK] Converted to grayscale (8-bit)")
    
    return img_resized, gray, image_id
