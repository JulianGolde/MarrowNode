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

## Experiment 02: Stage 2 Fine-Tuning & Uncertainty Estimation
**Date:** 2026-08-26 / 2026-08-27
**Action:** Unfroze MobileNetV3 backbone and applied a low learning rate (1e-4).
**Result:** Broke the 84% glass ceiling. The model is now highly accurate. 
**Architecture Update:** Implemented a Human-in-the-Loop safeguard in the Orchestrator. Any prediction with < 85% confidence is flagged as "Review Required" to mitigate the MLL dataset's weak labels (AI-generated ground truth).

## Data Engineering: Stage 1 (YOLOv8 Segmentation)
**Date:** 2026-08-27
**Dataset:** SegPC-2021 (ISBI Challenge).
**Action:** Built a robust conversion pipeline to transform clinical instance segmentation masks (.bmp) with multiple instances per image into YOLO normalized polygon format.

## Experiment 03: Stage 1 YOLOv8 Segmentation Training
**Date:** 2026-09-01
**Dataset:** SegPC-2021 (Clinical Ground Truth)
**Action:** Trained YOLOv8-Nano (yolov8n-seg.pt) for 50 epochs on the RTX 4080 Super.
**Result:** Phenomenal metrics for Edge Computing:
- **mAP50:** 86.8%
- **mAP50-95:** 77.5%
- **Inference Speed:** 0.9ms per image
**Architecture Update:** Replaced the mocked segmentation function in `orchestrator.py` with the live YOLOv8 model. The pipeline is now fully operational end-to-end.


## Experiment 04: Edge Optimization & ONNX Quantization (Phase 4)
**Date:** 2026-09-01
**Action:** Exported both Stage 1 (YOLOv8) and Stage 2 (MobileNetV3) models to ONNX format with FP16 (Half-Precision) quantization. 
**Architecture Update:** Completely rewrote the `MarrowPipeline` orchestrator to remove the heavy PyTorch dependency. Inference and image preprocessing are now handled exclusively by `onnxruntime` and pure `NumPy`.
**Result:** The system is now fully decoupled from GPU requirements and ready for low-resource CPU deployment. Latency benchmarks pending.
