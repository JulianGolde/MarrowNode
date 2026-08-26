"""
MarrowNode - Phase 2: Stage 2 Classifier Training
Architecture: MobileNetV3 (Small)
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration
DATA_DIR = "data/processed/mll_classification"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders():
    """Applies Data Augmentation to prevent overfitting and loads data."""
    # Transforms: Resize to MobileNet standard (224x224), add random flips for robustness
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes

def build_model(num_classes):
    """Loads a pre-trained Edge-optimized MobileNetV3 and adapts the final layer."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Freeze early layers to speed up training
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head for our specific number of classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model.to(DEVICE)

def train_model():
    logging.info(f"Starting training on device: {DEVICE}")
    train_loader, val_loader, class_names = get_data_loaders()
    logging.info(f"Detected classes: {class_names}")

    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the new classification head
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | Time: {epoch_time:.1f}s")

        # Save the best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/mobilenet_marrow_classifier.pth")
            
    logging.info(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")
    logging.info("Model saved to models/mobilenet_marrow_classifier.pth")

if __name__ == "__main__":
    train_model()