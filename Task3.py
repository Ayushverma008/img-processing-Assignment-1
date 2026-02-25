import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==================== TASK 3: IMAGE SAMPLING ====================
def analyze_sampling(gray_image):
    """
    Downsample to different resolutions and upsample back for comparison
    """
    print("\n--- TASK 3: SAMPLING ANALYSIS (Resolution Reduction) ---")
    
    resolutions = [512, 256, 128]
    labels = ["High (512x512)", "Medium (256x256)", "Low (128x128)"]
    sampled_images = []
    
    for i, (res, label) in enumerate(zip(resolutions, labels)):
        # Downsample
        downsampled = cv2.resize(gray_image, (res, res), interpolation=cv2.INTER_AREA)
        
        # Upsample back to 512x512 for visualization
        upsampled = cv2.resize(downsampled, (512, 512), interpolation=cv2.INTER_LINEAR)
        sampled_images.append(upsampled)
        
        # Save the downsampled version
        cv2.imwrite(f"outputs/sampled_{res}x{res}.png", downsampled)
        
        print(f"   [OK] {label}: {res}x{res} pixels (saved)")
    
    return sampled_images
