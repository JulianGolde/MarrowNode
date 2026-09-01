# MarrowNode: Offline Bone Marrow Segmentation System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Offline, full-stack decision-support ecosystem for instance segmentation and classification of bone marrow aspirates in low-resource clinical environments.

![MarrowNode Architecture Diagram](docs/images/architecture_diagram_placeholder.png)
*(Placeholder: Insert an architecture diagram showing the Stage 1 -> Stage 2 flow here)*

## Table of Contents
1. [Clinical Context & Architecture](#1-clinical-context--architecture)
2. [System Specifications](#2-system-specifications)
3. [Standard Operating Procedure (SOP)](#3-standard-operating-procedure-sop)
4. [Clinical Interface (Human-in-the-Loop)](#4-clinical-interface)
5. [Technology Stack](#5-technology-stack)
6. [Roadmap](#6-roadmap)

---

## 1. Clinical Context & Architecture

Bone marrow aspirate analysis presents extreme morphological complexity, hampered by dense cell occlusion and severe class imbalance across maturation stages. Existing automated solutions require prohibitive infrastructure, such as cloud computing and motorized digital microscopes. 

MarrowNode resolves this bottleneck by delivering a decentralized, edge-optimized software pipeline capable of operating entirely offline, processing optical microscopy images captured via standard consumer adapters.

## 2. System Architecture: The Two-Stage Pipeline

To overcome the medical annotation bottleneck and ensure long-term scalability, MarrowNode implements a decoupled, two-stage inference architecture. This modularity allows the system to scale to new pathologies without retraining the heavy segmentation backbone.

- **Stage 1: Agnostic Instance Segmentation (YOLOv8)**
  A lightweight segmentation model (YOLOv8-Nano) trained strictly on a binary class (Nucleated Cell vs. Background) using the SegPC-2021 clinical dataset. Its sole purpose is to handle occlusion, draw polygons, and locate targets within dense bone marrow smears.
  
- **The Bridge: Automated Isolation**
  An intermediary algorithmic step that computes the bounding box from the segmentation polygon, crops the raw image, and isolates the detected cell on a standardized background (masking out neighboring artifacts).

- **Stage 2: Expert Classification (MobileNetV3)**
  A highly optimized Convolutional Neural Network (MobileNetV3-Small) trained on isolated, expertly annotated single-cell datasets (such as the MLL dataset). It processes the cropped outputs from Stage 1 to classify the specific hematological lineage (e.g., Myeloblast, Monocyte, Lymphocyte).

- **Hardware Deployment Constraint:** Both stages, along with the color normalization pipeline (CLAHE), are engineered to run sequentially and offline on standard consumer hardware.
  
## 3. Standard Operating Procedure (SOP): MarrowNode Pipeline

### Objective
Establish a reproducible, scalable, and modular machine learning pipeline for bone marrow aspirate analysis, utilizing a two-stage approach (Segmentation -> Classification).

### Phase 1: Data Engineering & Standardization
1.  **Color Normalization Pipeline:**
    - Input: Raw RGB images from optical microscopes.
    - Process: Convert to LAB color space, apply CLAHE on the L-channel, convert back to RGB.
2.  **Dataset Preparation (Stage 2 - Classification):**
    - Ingest the MLL dataset (isolated cells). Apply color normalization.
3.  **Dataset Preparation (Stage 1 - Segmentation):**
    - Ingest the SegPC-2021 Challenge dataset.
    - Dynamically merge multi-instance `.bmp` masks and extract normalized YOLO polygon coordinates.

### Phase 2: Model Training & Validation
1.  **Train the Classifier (Stage 2):**
    - Architecture: MobileNetV3 (Fine-tuned, lr=1e-4).
    - Task: Multi-class classification.
2.  **Train the Segmenter (Stage 1):**
    - Architecture: YOLOv8-seg (Nano version).
    - Task: Binary instance segmentation on dense smears.

### Phase 3: Software Integration (Pipeline Orchestrator)
1.  **Develop the Bridge Script:** Calculate tight bounding boxes, mask ROI, and feed to the Classifier.
2.  **Safety Threshold (Uncertainty Estimation):** The orchestrator automatically flags any classification under 85% confidence as "Review Required" to prevent AI hallucinations on noisy clinical data.

## 4. Clinical Interface

The system features a lightweight, open-source Streamlit dashboard tailored for hematologists. It operates entirely locally, ensuring patient data privacy (no cloud processing).

![Streamlit Clinical Dashboard](docs/images/streamlit_ui_placeholder.png)
*(Placeholder: Insert a screenshot of the Streamlit dashboard processing a cell here)*

**Key Features:**
- **Dual Mode Inference:** Users can upload a Full Smear (triggers Stage 1 + 2) or an Isolated Cell (bypasses YOLO to prevent Scale Shift issues on cropped datasets).
- **Human-in-the-Loop:** Uncertain predictions are highlighted in red, forcing the clinician to make the final call.

## 5. Technology Stack
- **Deep Learning:** PyTorch, Ultralytics (YOLO)
- **Computer Vision:** OpenCV, Pillow
- **User Interface:** Streamlit (Customized for offline use)
- **Edge Deployment (WIP):** ONNX Runtime

## 6. Roadmap
- [X] Initialize repository and environment.
- [X] Configure data ingestion and color normalization scripts.
- [X] Establish Stage 2 classification model architecture (MobileNetV3).
- [X] Train Stage 1 instance segmentation model (YOLOv8) on clinical data.
- [X] Build and test the Human-in-the-loop clinical interface (Streamlit).
- [ ] Optimize models for Edge Computing (ONNX INT8 Quantization).

---
*Disclaimer: MarrowNode is an open-source Computer-Assisted Intervention (CAI) tool for research purposes. It does not provide autonomous medical diagnoses.*
