"""
MarrowNode - Phase 2: Inference Sanity Check
Tests the saved MobileNetV3 baseline model on a random hold-out image.
"""

import os
import random
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Configuration
MODEL_PATH = "models/mobilenet_marrow_classifier.pth"
TEST_DIR = "data/processed/mll_classification/test"
CLASSES = ["Lymphocyte", "Monocyte", "Myeloblast", "Neutrophil"]

def load_model(device):
    """Reconstructs the architecture and loads the saved weights."""
    model = models.mobilenet_v3_small(weights=None) # No need to download ImageNet weights again
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (shuts off dropout, etc.)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model on {device}...")
    model = load_model(device)
    
    # Preprocessing identical to validation phase
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Pick a random image from the test set
    true_class = random.choice(CLASSES)
    class_dir = os.path.join(TEST_DIR, true_class)
    image_name = random.choice(os.listdir(class_dir))
    image_path = os.path.join(class_dir, image_name)
    
    # Run Inference
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASSES[predicted_idx.item()]
    
    # Display Results
    print("\n" + "="*40)
    print("      MARROWNODE INFERENCE TEST")
    print("="*40)
    print(f"File: {image_name}")
    print(f"Ground Truth : {true_class}")
    print(f"Prediction   : {predicted_class}")
    print(f"Confidence   : {confidence.item() * 100:.2f}%")
    print("="*40)
    
    if true_class == predicted_class:
        print("[SUCCESS] The model predicted correctly.")
    else:
        print("[FAILED] The model made an error.")

if __name__ == "__main__":
    main()