"""
MarrowNode - Phase 1.2: FULL Classification Dataset Preparation (Big Data Ready)
Processes the entire dataset and streams directly to disk to prevent RAM overflow.
"""

import os
import cv2
import random
import numpy as np
import logging
from datasets import load_dataset
from PIL import ImageFile

# Prevent crashes on slightly corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

OUTPUT_BASE_DIR = "data/processed/mll_classification"
# Probability weights for Train/Val/Test
SPLIT_CHOICES = ["train", "val", "test"]
SPLIT_WEIGHTS = [0.70, 0.15, 0.15] 

# Target classes
TARGET_CLASSES = {
    0: "Erythroblast",
    1: "Lymphocyte", 
    2: "Monocyte",
    3: "Myeloblast",
    4: "Neutrophil"
}

def normalize_giemsa(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def setup_directories():
    for split in SPLIT_CHOICES:
        for class_name in TARGET_CLASSES.values():
            os.makedirs(os.path.join(OUTPUT_BASE_DIR, split, class_name), exist_ok=True)

def main():
    setup_directories()
    logging.info("Directories ready. Loading full MLL dataset...")
    
    # We load the full dataset (it will cache locally)
    dataset = load_dataset("ekim15/bone_marrow_cell_dataset", split="train")
    
    class_counts = {class_name: 0 for class_name in TARGET_CLASSES.values()}
    processed_total = 0
    
    logging.info(f"Total images in dataset: {len(dataset)}. Starting processing...")

    for idx, sample in enumerate(dataset):
        label_id = sample["label"]
        if label_id not in TARGET_CLASSES:
            continue
            
        class_name = TARGET_CLASSES[label_id]
        
        try:
            # 1. Process image
            raw_image = np.array(sample["image"])
            norm_image = normalize_giemsa(raw_image)
            norm_bgr = cv2.cvtColor(norm_image, cv2.COLOR_RGB2BGR)
            
            # 2. Randomly assign to train, val, or test on the fly
            split = random.choices(SPLIT_CHOICES, weights=SPLIT_WEIGHTS, k=1)[0]
            
            # 3. Save directly to SSD
            filename = f"{class_name}_{processed_total}.jpg"
            filepath = os.path.join(OUTPUT_BASE_DIR, split, class_name, filename)
            cv2.imwrite(filepath, norm_bgr)
            
            class_counts[class_name] += 1
            processed_total += 1
            
        except Exception as e:
            logging.warning(f"Skipping corrupted image at index {idx}: {e}")
            continue

        # Log progress every 5000 images
        if processed_total % 5000 == 0:
            logging.info(f"Processed {processed_total} images... Current distribution: {class_counts}")
                
    logging.info("Extraction complete!")
    logging.info(f"Final distribution: {class_counts}")

if __name__ == "__main__":
    main()