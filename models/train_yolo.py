"""
MarrowNode - Phase 3.2: YOLOv8 Segmentation Training
Trains a YOLOv8 Nano model on the SegPC-2021 clinical dataset.
Optimized for Edge Computing (high speed, low memory).
"""

import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_segmentation_model():
    logging.info("Initializing YOLOv8-Nano Segmentation model...")
    
    # On charge le modèle "Nano" (le plus léger et rapide) pré-entraîné
    model = YOLO('yolov8n-seg.pt')
    
    logging.info("Starting training on RTX 4080 Super...")
    
    # Lancement de l'entraînement
    results = model.train(
        data='data/processed/yolo_dataset/dataset.yaml',
        epochs=50,                  
        imgsz=640,                  
        batch=16,                   
        device=0,                   
        project='models',           
        name='yolo_marrow_seg',     
        exist_ok=True,              
        patience=10                 
    )
    
    logging.info("YOLOv8 Training Complete! Best weights saved in: models/yolo_marrow_seg/weights/best.pt")

if __name__ == "__main__":
    train_segmentation_model()