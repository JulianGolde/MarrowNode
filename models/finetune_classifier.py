"""
MarrowNode - Phase 2.1: Model Fine-Tuning
Unfreezes the entire MobileNetV3 architecture to break the accuracy glass ceiling.
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
BASE_MODEL_PATH = "models/mobilenet_marrow_classifier.pth"
FINETUNED_MODEL_PATH = "models/mobilenet_marrow_finetuned.pth"
BATCH_SIZE = 32
EPOCHS = 10
# VERY IMPORTANT: Low learning rate to avoid Catastrophic Forgetting
LEARNING_RATE = 1e-4 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders():
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

def load_base_model(num_classes):
    """Loads the previously trained baseline model and UNFREEZES all layers."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    # Load the baseline weights we trained earlier
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    
    # UNFREEZE ALL LAYERS FOR FINE-TUNING
    for param in model.parameters():
        param.requires_grad = True
        
    return model.to(DEVICE)

def fine_tune_model():
    logging.info(f"Starting Fine-Tuning on device: {DEVICE}")
    train_loader, val_loader, class_names = get_data_loaders()
    
    model = load_base_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    
    # Optimize ALL parameters now, but with a tiny learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start tracking from our assumed baseline accuracy
    best_acc = 0.84 

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
        logging.info(f"FT Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | Time: {epoch_time:.1f}s")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            logging.info(f"--> New best model saved with accuracy: {best_acc:.4f}")
            
    logging.info(f"Fine-Tuning Complete. Peak Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    fine_tune_model()