# SmartYard Project Roadmap: Kaleris AI/ML Application

This document outlines the strategic phases to transform SmartYard into a production-grade Yard Management AI system, specifically tailored for an AI/ML Engineer application at Kaleris.

---

## 🏗️ Phase 1 — System Architecture & Baseline MVP (Current)
**Goal:** Establish the engineering foundation and demonstrate an end-to-end pipeline ready for resume benchmarking.

### Key Objectives:
1.  **Environment Stability**: Deploy virtual environment and install core ML dependencies (Torch, Ultralytics, OpenCV).
2.  **Directory Infrastructure**: Implement enterprise-standard data and model hierarchy.
3.  **Baseline Engine**: Download and integrate a baseline YOLOv8s model to ensure the pipeline is immediately functional.
4.  **Hardware Benchmarking**: Measure and document inference latency (ms/image) on RTX 3050 hardware.
5.  **Analytics Integration**: Hooking up the detection pipeline to **Gate Logging**, **Zone Mapping**, and **Anomaly Detection** modules.
6.  **Kaleris Dashboard**: Initial launch of the Streamlit dashboard for real-time asset visibility.

**Phase 1 Resume Impact:**
- Demonstrates **Systems Design** and **Pipeline Engineering** capabilities.
- Provides concrete performance numbers (FPS, Latency) for technical interviews.

---

## 📊 Phase 2 — Dataset Ingestion & Engineering (Coming Next)
**Goal:** Scale the system to handle massive logistics-specific datasets.

### Key Objectives:
1.  **Roboflow Integration**: Automated download of the **100k+ image Logistics dataset**.
2.  **Data Cleaning & Filtering**: Implementing logic to handle the 8 specific logistics classes (Truck, Container, Forklift, Person, Safety Gear).
3.  **Preprocessing Pipeline**: Image resizing (640x640), normalization, and YOLO-format label verification.
4.  **Augmentation Strategy**: Designing augmentations (Mosaic, Blur, Mixup) to handle varying yard lighting conditions.

---

## 🧠 Phase 3 — Domain-Specific Fine-Tuning
**Goal:** Train the specialized "SmartYard" model for high-precision logistics detection.

### Key Objectives:
1.  **Transfer Learning**: Fine-tune YOLOv8s on the combined logistics and truck-container datasets.
2.  **Hyperparameter Optimization**: Tune learning rates, batch sizes, and optimizer settings for 4GB VRAM.
3.  **Safety Compliance Training**: Specifically optimize for high recall on PPE detection (Helmets, Vests).

---

## 📈 Phase 4 — Evaluation, Optimization & Export
**Goal:** Prove the model's reliability and prepare for production deployment.

### Key Objectives:
1.  **Rigorous Evaluation**: Generate mAP@50, Precision-Recall curves, and Confusion Matrices.
2.  **Optimization**: Explore model quantization (FP16/INT8) or TensorRT export for edge deployment.
3.  **Kaleris Final Showcase**: Polish the dashboard, update README with final metrics, and package the repository for submission.

---

## 📝 Future Use Cases (Post-MVP)
- **Container Number Recognition (OCR)**: Automating the reading of Container IDs.
- **Congestion Analysis**: Using heatmaps to predict yard bottlenecks.
- **Drone-Based Inspections**: Adapting the model for top-down yard imaging.
