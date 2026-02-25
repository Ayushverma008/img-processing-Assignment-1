

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime



# ==================== TASK 4: IMAGE QUANTIZATION ====================
def quantize_image(gray_image, levels):
    """
    Reduce number of gray levels
    """
    step = 256 // levels
    quantized = (gray_image // step) * step
    return quantized.astype(np.uint8)

def analyze_quantization(gray_image):
    """
    Quantize to different bit depths
    """
    print("\n--- TASK 4: QUANTIZATION ANALYSIS (Bit-depth Reduction) ---")
    
    # Original is already 8-bit (256 levels)
    bit_depths = [8, 4, 2]
    gray_levels = [256, 16, 4]
    labels = ["8-bit (256 levels)", "4-bit (16 levels)", "2-bit (4 levels)"]
    quantized_images = []
    
    # Original 8-bit image
    quantized_images.append(gray_image)
    cv2.imwrite("outputs/quantized_8bit.png", gray_image)
    print(f"   [OK] {labels[0]} (saved)")
    
    # 4-bit quantization
    q_16 = quantize_image(gray_image, 16)
    quantized_images.append(q_16)
    cv2.imwrite("outputs/quantized_4bit.png", q_16)
    print(f"   [OK] {labels[1]} (saved)")
    
    # 2-bit quantization
    q_4 = quantize_image(gray_image, 4)
    quantized_images.append(q_4)
    cv2.imwrite("outputs/quantized_2bit.png", q_4)
    print(f"   [OK] {labels[2]} (saved)")
    
    return quantized_images
