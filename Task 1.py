"""
Name       : [Ayush verma]
Roll No    : [2301010447]
Course     : Image Processing & Computer Vision
Unit       : Unit 1 - Image Fundamentals
Assignment : Smart Document Scanner & Quality Analysis System
Date       : [25/02/26]
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==================== TASK 1: PROJECT SETUP ====================
print("="*70)
print("           SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM")
print("           Course: Image Processing & Computer Vision")
print("           Simulates image acquisition, sampling & quantization effects")
print("="*70)
print(f"Run Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Create outputs directory if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")
    print("\n[OK] Created 'outputs' directory for saving results")
