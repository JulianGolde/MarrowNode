# MarrowNode: Offline Bone Marrow Segmentation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Offline, full-stack decision-support ecosystem for instance segmentation of bone marrow aspirates in low-resource environments.

## Table of Contents
1. [Clinical Context & Architecture](#clinical-context--architecture)
2. [System Specifications](#system-specifications)
3. [Standard Operating Procedure (SOP)](#standard-operating-procedure-sop)
4. [Technology Stack](#technology-stack)
5. [Roadmap](#roadmap)

---

## 1. Clinical Context & Architecture

Bone marrow aspirate analysis presents extreme morphological complexity, hampered by dense cell occlusion and severe class imbalance across maturation stages. Existing automated solutions require prohibitive infrastructure, such as cloud computing and motorized digital microscopes. 

MarrowNode resolves this bottleneck by delivering a decentralized, edge-optimized software pipeline capable of operating entirely offline, processing optical microscopy images captured via standard consumer adapters.

## 2. System Architecture: The Two-Stage Pipeline

To overcome the medical annotation bottleneck and ensure long-term scalability, MarrowNode implements a decoupled, two-stage inference architecture. This modularity allows the system to scale to new pathologies without retraining the heavy segmentation backbone.

- **Stage 1: Agnostic Instance Segmentation (The Finder)**
  A lightweight segmentation model (e.g., YOLOv8-seg) trained strictly on a binary class (Nucleated Cell vs. Background). Its sole purpose is to handle occlusion, draw polygons, and locate targets within dense bone marrow smears.
  
- **The Bridge: Automated Isolation**
  An intermediary algorithmic step that computes the bounding box from the segmentation polygon, crops the raw image, and isolates the detected cell on a standardized background (masking out neighboring artifacts).

- **Stage 2: Expert Classification (The Classifier)**
  A highly optimized Convolutional Neural Network (e.g., MobileNetV3 or ResNet) trained on isolated, expertly annotated single-cell datasets (such as the MLL dataset). It processes the cropped outputs from Stage 1 to classify the specific hematological lineage (e.g., Myeloblast, Promyelocyte, Lymphocyte).

- **Hardware Deployment Constraint:** Both stages, along with the color normalization pipeline (CLAHE), are engineered to run sequentially and offline on standard consumer hardware via INT8 quantization.
  
## 3. # Standard Operating Procedure (SOP): MarrowNode Pipeline

## Objective
Establish a reproducible, scalable, and modular machine learning pipeline for bone marrow aspirate analysis, utilizing a two-stage approach (Segmentation -> Classification).

## Phase 1: Data Engineering & Standardization
1.  **Color Normalization Pipeline:**
    - Input: Raw RGB images from optical microscopes.
    - Process: Convert to LAB color space, apply CLAHE on the L-channel, convert back to RGB.
    - Output: Illumination-standardized images.
2.  **Dataset Preparation (Stage 2 - Classification):**
    - Ingest the MLL dataset (isolated cells).
    - Apply color normalization.
    - Split dataset into Train (70%), Validation (15%), and Test (15%).
3.  **Dataset Preparation (Stage 1 - Segmentation):**
    - Acquire raw bone marrow smear images (dense clusters).
    - Annotate a binary class (Nucleated Cell) using bounding polygons.
    - Format annotations to YOLO-seg standard.

## Phase 2: Model Training & Validation
1.  **Train the Classifier (Stage 2):**
    - Architecture: MobileNetV3 (optimized for edge).
    - Task: Multi-class classification on the normalized MLL dataset.
    - Success Metric: F1-Score > 0.90 on the Test set.
2.  **Train the Segmenter (Stage 1):**
    - Architecture: YOLOv8-seg (Nano or Small version).
    - Task: Binary instance segmentation on dense smears.
    - Success Metric: mIoU (Mean Intersection over Union) > 0.85.

## Phase 3: Software Integration (Pipeline Orchestrator)
1.  **Develop the Bridge Script:**
    - Receive polygon coordinates from the Segmenter.
    - Calculate tight bounding boxes.
    - Crop and mask the region of interest (ROI) from the original image.
    - Feed the isolated ROI to the Classifier.
2.  **Object-Oriented Encapsulation:**
    - Wrap the entire sequential logic (Normalization -> Segmenter -> Cropper -> Classifier) into a single, robust Python class (`MarrowPipeline`).

## Phase 4: Edge Optimization & UI
1.  **Quantization:** Convert PyTorch model weights (`.pt`) to ONNX Runtime format utilizing FP16 or INT8 precision.
2.  **Latency Benchmarking:** Ensure end-to-end inference time is under 2000ms per full microscopic field on CPU.
3.  **Interface Deployment:** Launch the local web interface (Streamlit/FastAPI) to allow clinical users to upload images and review predictions.

## 4. Technology Stack

- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Scikit-Image
- **Edge Deployment:** ONNX Runtime, TensorRT
- **Networking:** PySerial, Socket (Python standard library)
- **Interface:** FastAPI, Streamlit

## 5. Roadmap

- [ X] Initialize repository and environment.
- [X ] Configure data ingestion and color normalization scripts.
- [ X] Xstablish baseline segmentation model architecture.
- [ ] IXplement local network ingestion protocols.
- [ ] Build and test the Human-in-the-loop clinical interface.

# MarrowNode: Experiment & Training Log

## Experiment 01: Stage 2 Classifier Baseline (Transfer Learning)
**Date:** 2026-08-26
**Objective:** Establish a performance baseline for the Stage 2 classification module using a frozen pre-trained architecture.

### Hardware & Environment
- **GPU:** NVIDIA RTX 4080 Super (16GB VRAM)
- **CPU:** AMD Ryzen 7 5800X
- **Framework:** PyTorch (CUDA enabled)

### Dataset Configuration
- **Source:** MLL Bone Marrow Morphology Dataset
- **Classes (4):** Lymphocyte, Monocyte, Myeloblast, Neutrophil
- **Distribution:** ~15,000 images per class (Total: ~60,000)
- **Split:** Train (70%), Validation (15%), Test (15%)
- **Preprocessing:** CLAHE on LAB color space (Giemsa normalization).

### Model Architecture & Hyperparameters
- **Backbone:** MobileNetV3-Small (Weights: IMAGENET1K_V1)
- **Strategy:** Transfer Learning (Feature extractor frozen, only the final linear classification head is active).
- **Batch Size:** 32
- **Learning Rate:** 0.001 (Adam Optimizer)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10

### Results & Metrics
- **Total Training Time:** ~21 minutes (~125s per epoch).
- **Peak Performance:** Epoch 5
  - **Train Loss:** 0.4499 | **Train Accuracy:** 0.8277
  - **Val Loss:** 0.4308 | **Val Accuracy:** 0.8438

### Engineering Notes & Analysis
The model achieved an 84.38% validation accuracy by merely tuning the final classification head, demonstrating the high quality of the color normalization pipeline and the robustness of the dataset. 
The slight oscillation in validation metrics (Epoch 6 to 10) indicates that the frozen feature extractor has reached its representational limit (the "glass ceiling" of transfer learning). 
**Next Step:** Implement a full-network fine-tuning phase with a reduced learning rate to surpass the 90% accuracy threshold.
---
*Disclaimer: MarrowNode is a Computer-Assisted Intervention (CAI) tool for research purposes. It does not provide autonomous medical diagnoses.*
