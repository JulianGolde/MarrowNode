"""
MarrowNode - Core Pipeline Orchestrator (Final)
Integrates Stage 1 (YOLOv8 Segmentation) and Stage 2 (MobileNetV3 Classification).
Features Human-in-the-loop thresholding for clinical safety.
"""

import os
import cv2
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from typing import List, Dict, Any
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarrowPipeline:
    def __init__(self, yolo_path: str, classifier_path: str, safety_threshold: float = 0.85):
        self.safety_threshold = safety_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Lymphocyte", "Monocyte", "Myeloblast", "Neutrophil"]
        
        logging.info(f"Initializing MarrowNode Pipeline on {self.device}...")
        
        # 1. Load Preprocessing Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 2. Load Models
        self.yolo_model = YOLO(yolo_path)
        self.classifier = self._load_classifier(classifier_path)
        logging.info("Both Stage 1 and Stage 2 models loaded successfully.")

    def _load_classifier(self, path: str) -> nn.Module:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier weights not found at {path}")
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(self.classes))
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _normalize_image(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _run_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """STAGE 1: Detects and extracts precise cell polygons using YOLOv8."""
        results = self.yolo_model(image, verbose=False)
        polygons = []
        
        if len(results) > 0 and results[0].masks is not None:
            # YOLO returns coordinates. We convert them to integers for OpenCV.
            for mask_coords in results[0].masks.xy:
                if len(mask_coords) > 0:
                    poly = np.array(mask_coords, dtype=np.int32)
                    polygons.append(poly)
                    
        return polygons

    def _isolate_cells(self, image: np.ndarray, polygons: List[np.ndarray]) -> List[np.ndarray]:
        """Masks everything outside the polygon and crops the bounding box."""
        isolated_crops = []
        for poly in polygons:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            x, y, w, h = cv2.boundingRect(poly)
            # Add a tiny padding to avoid clipping the edges of the cell
            padding = 5
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
            
            crop = masked_image[y1:y2, x1:x2]
            # Ensure crop is not empty before adding
            if crop.size > 0:
                isolated_crops.append(crop)
            
        return isolated_crops

    def _run_classification(self, crops: List[np.ndarray]) -> List[Dict[str, Any]]:
        """STAGE 2: Classifies each cell crop."""
        results = []
        if not crops:
            return results
            
        with torch.no_grad():
            for crop in crops:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                output = self.classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                conf_score = confidence.item()
                raw_class = self.classes[predicted_idx.item()]
                
                if conf_score < self.safety_threshold:
                    final_class = "Review Required (Uncertain)"
                    flagged = True
                else:
                    final_class = raw_class
                    flagged = False
                
                results.append({
                    "predicted_class": final_class,
                    "raw_prediction": raw_class,
                    "confidence": round(conf_score * 100, 2),
                    "flagged_for_review": flagged
                })
                
        return results

    def process_image(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        logging.info(f"Processing image: {image_path}")
        
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")
            
        norm_image = self._normalize_image(raw_image)
        polygons = self._run_segmentation(norm_image)
        crops = self._isolate_cells(norm_image, polygons)
        classifications = self._run_classification(crops)
        
        latency_ms = (time.time() - start_time) * 1000
        logging.info(f"Pipeline executed in {latency_ms:.2f} ms")
        
        return {
            "status": "success",
            "latency_ms": round(latency_ms, 2),
            "cells_detected": len(polygons),
            "results": classifications
        }

if __name__ == "__main__":
    # Test file from our SegPC validation set
    test_img = "data/raw/segpc/valid/x/106.bmp"
    
    if os.path.exists(test_img):
        # Note the path to the YOLO weights generated in your terminal
        pipeline = MarrowPipeline(
            yolo_path="runs/segment/models/yolo_marrow_seg/weights/best.pt", 
            classifier_path="models/mobilenet_marrow_finetuned.pth", 
            safety_threshold=0.85
        )
        report = pipeline.process_image(test_img)
        
        print("\n" + "="*40)
        print("      FINAL PIPELINE REPORT")
        print("="*40)
        import json
        print(json.dumps(report, indent=4))
    else:
        print(f"Please provide a valid image path to test.")