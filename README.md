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

## 2. System Specifications

- **Local Inference:** Offline execution on consumer-grade x86/ARM hardware utilizing INT8/FP16 model quantization via ONNX Runtime and TensorRT.
- **Multi-Protocol Ingestion:** Modular communication pipeline accepting incoming diagnostic images via direct USB (UVC protocol), Bluetooth Low Energy (BLE), and ad-hoc local Wi-Fi networks (air-gapped).
- **Instance Segmentation:** Advanced CNN architecture (e.g., YOLO-seg / Mask R-CNN) engineered to resolve dense object occlusion and overlapping cellular boundaries.
- **Class Imbalance Optimization:** Implementation of specialized loss functions (e.g., Focal Loss, Dice Loss) to address the distribution skew between prevalent erythrocytes and rare nucleated lineages.
- **Human-in-the-Loop Integration:** Local web application enabling clinical validation, manual mask correction, and automated Myeloid-to-Erythroid (M:E) ratio calculation in compliance with WHO guidelines.

## 3. Standard Operating Procedure (SOP)

The development lifecycle follows a strict machine learning engineering protocol:

### Phase 1: Data Engineering
- **Acquisition:** Aggregation of public bone marrow aspirate datasets.
- **Preprocessing:** Application of color deconvolution algorithms to normalize Giemsa stain variations across different illumination conditions.
- **Splitting:** Strict enforcement of Train/Validation/Hold-out Test splits to prevent data leakage.

### Phase 2: Model Training
- **Hardware:** Local training on NVIDIA RTX 4080 Super (16GB VRAM).
- **Optimization:** Hyperparameter tuning for learning rate, batch size, and loss function weights.
- **Metrics:** Evaluation using Mean Intersection over Union (mIoU) and mean Average Precision (mAP).

### Phase 3: Edge Optimization
- **Quantization:** Post-training quantization (PTQ) to reduce memory footprint.
- **Benchmarking:** CPU/Edge GPU inference latency stress testing.

### Phase 4: System Integration
- **Backend:** Implementation of multi-protocol network listeners (USB/BT/WLAN).
- **Frontend:** Deployment of the clinical validation GUI.

## 4. Technology Stack

- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Scikit-Image
- **Edge Deployment:** ONNX Runtime, TensorRT
- **Networking:** PySerial, Socket (Python standard library)
- **Interface:** FastAPI, Streamlit

## 5. Roadmap

- [ ] Initialize repository and environment.
- [ ] Configure data ingestion and color normalization scripts.
- [ ] Establish baseline segmentation model architecture.
- [ ] Implement local network ingestion protocols.
- [ ] Build and test the Human-in-the-loop clinical interface.

---
*Disclaimer: MarrowNode is a Computer-Assisted Intervention (CAI) tool for research purposes. It does not provide autonomous medical diagnoses.*
