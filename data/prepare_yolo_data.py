"""
MarrowNode - Phase 3.1: YOLOv8 Data Preparation (Absolute Thresholding)
Converts SegPC-2021 binary masks into YOLOv8 Polygon format.
Fixes the low-intensity pixel issue found in medical instance masks.
"""

import os
import cv2
import glob
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAW_DATA_DIR = "data/raw/segpc"
YOLO_DIR = "data/processed/yolo_dataset"

def create_yolo_structure():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

def process_masks_to_polygons(split_name: str, raw_split_folder: str):
    img_dir = os.path.join(RAW_DATA_DIR, raw_split_folder, 'x')
    mask_dir = os.path.join(RAW_DATA_DIR, raw_split_folder, 'y')
    
    yolo_split = 'val' if split_name == 'valid' else split_name
    out_img_dir = os.path.join(YOLO_DIR, 'images', yolo_split)
    out_label_dir = os.path.join(YOLO_DIR, 'labels', yolo_split)
    
    if not os.path.exists(img_dir):
        return

    processed_count = 0
    all_images = os.listdir(img_dir)
    logging.info(f"[{split_name}] Found {len(all_images)} files in folder 'x'. Processing...")

    for img_file in all_images:
        base_name = img_file.split('.')[0] 
        img_path = os.path.join(img_dir, img_file)
        
        mask_pattern = os.path.join(mask_dir, f"{base_name}_*.bmp")
        mask_files = glob.glob(mask_pattern)
        
        if not mask_files:
            mask_files = glob.glob(os.path.join(mask_dir, f"{base_name}.bmp"))
            
        if not mask_files:
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        polygons = []
        
        for mask_path in mask_files:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # LE FIX EST ICI : Tout pixel supérieur à 0 devient blanc (255)
            _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                    
                coords = []
                for point in contour:
                    x = point[0][0] / w
                    y = point[0][1] / h
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])
                
                yolo_line = "0 " + " ".join(coords)
                polygons.append(yolo_line)
            
        if polygons:
            label_file = os.path.join(out_label_dir, f"{base_name}.txt")
            with open(label_file, "w") as f:
                f.write("\n".join(polygons))
                
            out_img_path = os.path.join(out_img_dir, f"{base_name}.jpg")
            cv2.imwrite(out_img_path, img)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                logging.info(f"--> Converted {processed_count} images so far...")

    logging.info(f"Successfully processed {processed_count} images for {split_name} split.")

def generate_yaml():
    yaml_content = f"""path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val

names:
  0: Cell
"""
    with open(os.path.join(YOLO_DIR, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    create_yolo_structure()
    process_masks_to_polygons('train', 'train')
    process_masks_to_polygons('valid', 'valid')
    generate_yaml()
    logging.info("YOLO dataset is ready!")