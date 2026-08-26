"""
MarrowNode - Phase 1.1: Data Acquisition & Color Normalization
Target: MLL Bone Marrow Morphology Dataset (HuggingFace)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# Directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalize_giemsa_stain(image_rgb):
    """
    Standardizes Giemsa-stained microscopic images.
    Converts to LAB color space and applies CLAHE to the Luminance channel
    to correct uneven microscope illumination while preserving cell colors.
    """
    # Convert RGB to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge back and convert to RGB
    limg = cv2.merge((cl, a_channel, b_channel))
    normalized_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return normalized_rgb

def main():
    print("[INFO] Fetching MLL Bone Marrow Dataset from HuggingFace...")
    # Downloading a small subset (streaming mode to avoid downloading 7GB instantly)
    dataset = load_dataset("ekim15/bone_marrow_cell_dataset", split="train", streaming=True)
    
    print("[INFO] Processing and Normalizing first 5 samples...")
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 15))
    axes[0, 0].set_title("Raw Image (Microscope)")
    axes[0, 1].set_title("Normalized (CLAHE LAB)")

    for i, sample in enumerate(dataset):
        if i >= 5:
            break
            
        # Extract PIL Image and convert to NumPy array (RGB)
        raw_image = np.array(sample["image"])
        label = sample["label"] # Clinical class
        
        # Apply normalization algorithm
        norm_image = normalize_giemsa_stain(raw_image)
        
        # Save processed data
        cv2.imwrite(os.path.join(PROCESSED_DIR, f"norm_cell_{i}_class_{label}.jpg"), cv2.cvtColor(norm_image, cv2.COLOR_RGB2BGR))
        
        # Visualization
        axes[i, 0].imshow(raw_image)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(norm_image)
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("data/normalization_benchmark.png", dpi=300)
    print("[SUCCESS] Processing complete. Benchmark saved to data/normalization_benchmark.png")

if __name__ == "__main__":
    main()