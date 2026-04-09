# 🚛 SmartYard — AI-Powered Yard Asset Detection & Management System

> Built with YOLOv8 + OpenCV | Inspired by real-world Yard Management Systems (YMS) used in logistics & supply chain

---

## 📌 Project Overview

SmartYard is an end-to-end computer vision system that detects and classifies logistics assets (trucks, containers, forklifts) in yard images, maps them to defined zones, generates occupancy reports, flags anomalies, and logs gate entries — all from static images.

This project mirrors core functionality of enterprise Yard Management Systems used by companies like Kaleris, simulating real-world supply chain intelligence using open-source tools.

---

## 🎯 Use Cases Built On Top of Detection

| Use Case                     | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| 🔍 Asset Detection           | Detect trucks, containers, forklifts, persons in yard images          |
| 🗺️ Zone Occupancy Mapping  | Map assets to defined yard zones, calculate occupancy %               |
| 🚨 Anomaly Alert System      | Flag unauthorized vehicles, zone violations, restricted area breaches |
| 🚪 Automated Gate Entry Log  | Auto-log detected vehicles at gate with timestamp + ID                |
| 📊 Occupancy Heatmap         | Aggregate detections across images to show congestion patterns        |
| ✅ Safety Compliance Checker | Check if persons have helmets/vests in yard images                    |
| 📁 Report Generation         | Export structured JSON + CSV reports per image                        |
| 🖥️ Streamlit Dashboard     | Interactive UI to upload images, view detections, download reports    |

---

## 🗂️ Project Structure

```
smartyard/
├── data/
│   ├── raw/                    # Downloaded dataset images
│   ├── processed/              # Resized and cleaned images
│   ├── annotations/            # YOLO format label files (.txt)
│   └── test_images/            # Images used for final demo
│
├── models/
│   └── best.pt                 # Fine-tuned YOLOv8 weights
│
├── outputs/
│   ├── annotated/              # Output images with bounding boxes
│   ├── reports/                # JSON and CSV reports per image
│   └── heatmaps/               # Zone occupancy heatmap images
│
├── src/
│   ├── preprocess.py           # Dataset download, cleaning, formatting
│   ├── train.py                # YOLOv8 fine-tuning script
│   ├── evaluate.py             # mAP, confusion matrix, precision/recall
│   ├── detect.py               # Run inference on images
│   ├── annotate.py             # OpenCV bounding boxes, zone overlays
│   ├── zone_mapper.py          # Define zones, map detections to zones
│   ├── anomaly.py              # Rule-based anomaly and alert engine
│   ├── gate_logger.py          # Automated gate entry logging
│   ├── heatmap.py              # Aggregate detections into heatmap
│   ├── compliance.py           # Safety compliance checker
│   ├── report.py               # JSON + CSV report generator
│   └── dashboard.py            # Streamlit dashboard app
│
├── config/
│   ├── dataset.yaml            # YOLO dataset config (classes, paths)
│   └── zones.json              # Zone definitions (coordinates per image)
│
├── requirements.txt
├── main.py                     # Entry point — runs full pipeline
└── README.md
```

---

## ⚙️ Tech Stack

| Component             | Tool                 |
| --------------------- | -------------------- |
| Object Detection      | YOLOv8 (Ultralytics) |
| Image Processing      | OpenCV               |
| Dataset Management    | Roboflow Universe    |
| Dashboard             | Streamlit            |
| Data Handling         | Pandas, NumPy        |
| Visualization         | Matplotlib, Seaborn  |
| Deep Learning Backend | PyTorch (CUDA)       |
| Report Export         | JSON, CSV            |
| Version Control       | Git + GitHub         |

---

## 📦 Datasets Used

| Dataset                    | Source                   | Images | Key Classes                                                     |
| -------------------------- | ------------------------ | ------ | --------------------------------------------------------------- |
| Logistics Object Detection | Roboflow Universe        | 99,238 | truck, forklift, freight container, person, helmet, safety vest |
| Yard Management System     | Roboflow Universe        | 157    | Chassis ID, Container ID                                        |
| Truck Container Dataset    | Roboflow Universe        | 4,703  | truck, container                                                |
| Forklift Detection         | HuggingFace (keremberke) | —     | Pretrained YOLOv8s model                                        |

> **Strategy:** Start from the Roboflow Logistics pretrained model (mAP 76%) and fine-tune on the YMS + Truck Container datasets for domain-specific accuracy.

---

## 🔢 Implementation Plan — Phase by Phase

---

### ✅ Phase 1 — Environment Setup *(Day 1)*

**Goal:** Get your environment ready to run everything.

**Steps:**

1. Create a virtual environment
   ```bash
   python -m venv smartyard_env
   source smartyard_env/bin/activate        # Linux/Mac
   smartyard_env\Scripts\activate           # Windows
   ```
2. Install dependencies
   ```bash
   pip install ultralytics opencv-python numpy pandas matplotlib seaborn streamlit roboflow torch torchvision
   ```
3. Verify GPU setup
   ```python
   import torch
   print(torch.cuda.is_available())         # Should print True
   print(torch.cuda.get_device_name(0))     # Should show your GPU
   ```
4. Create project folder structure
   ```bash
   mkdir -p smartyard/{data/{raw,processed,annotations,test_images},models,outputs/{annotated,reports,heatmaps},src,config}
   ```

**Deliverable:** Working environment with GPU confirmed ✔

---

### ✅ Phase 2 — Dataset Download & Preparation *(Day 2)*

**Goal:** Get the right data in the right format.

**Steps:**

1. Sign up at [roboflow.com](https://roboflow.com/) and get a free API key
2. Download Logistics dataset via Roboflow Python API
   ```python
   # src/preprocess.py
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("large-benchmark-datasets").project("logistics-sz9jr")
   dataset = project.version(1).download("yolov8")
   ```
3. Download Yard Management System dataset
   ```python
   project = rf.workspace("roboflow-universe-projects").project("yard-management-system")
   dataset = project.version(1).download("yolov8")
   ```
4. Download Truck Container dataset
   ```python
   project = rf.workspace("data-woimc").project("truck-container-q4wrp")
   dataset = project.version(3).download("yolov8")
   ```
5. Filter and keep only relevant classes:
   * `truck`, `trailer`, `container`, `freight_container`, `forklift`, `person`, `helmet`, `safety_vest`
6. Merge all datasets into one unified folder
7. Resize all images to `640x640`
8. Split into train/val/test → **70% / 20% / 10%**
9. Create `config/dataset.yaml`
   ```yaml
   path: ./data/processed
   train: train/images
   val: val/images
   test: test/images
   nc: 8
   names: ['truck', 'trailer', 'container', 'forklift', 'person', 'helmet', 'safety_vest', 'car']
   ```

**Deliverable:** Clean, formatted dataset in YOLO structure ✔

---

### ✅ Phase 3 — Model Fine-Tuning *(Day 3–4)*

**Goal:** Train a domain-specific model on logistics/yard data.

**Steps:**

1. Start from Roboflow Logistics pretrained weights (mAP 76% baseline)
2. Fine-tune with YOLOv8s (small — optimal for your 4GB GPU)
   ```python
   # src/train.py
   from ultralytics import YOLO

   model = YOLO('yolov8s.pt')   # or load Roboflow pretrained weights

   model.train(
       data='config/dataset.yaml',
       epochs=50,
       imgsz=640,
       batch=8,            # Safe for 4GB GPU
       device=0,           # Use GPU
       patience=10,        # Early stopping
       save=True,
       project='models',
       name='smartyard_v1'
   )
   ```
3. Monitor training:
   * Watch `train/box_loss` and `val/box_loss` curves
   * Ensure val loss decreases (no overfitting)
   * Target **mAP@50 > 0.70**
4. Best weights auto-saved to `models/smartyard_v1/weights/best.pt`
   * Copy to `models/best.pt`

**Training Tips:**

* If GPU runs out of memory → reduce `batch` to 4
* If mAP is low → increase epochs to 100
* If overfitting → add `augment=True` in training args

**Deliverable:** `models/best.pt` with mAP > 0.70 ✔

---

### ✅ Phase 4 — Model Evaluation *(Day 4)*

**Goal:** Verify your model is reliable before building on top of it.

**Steps:**

1. Run validation on test set
   ```python
   # src/evaluate.py
   from ultralytics import YOLO
   model = YOLO('models/best.pt')
   metrics = model.val(data='config/dataset.yaml', split='test')
   print(f"mAP@50: {metrics.box.map50}")
   print(f"Precision: {metrics.box.mp}")
   print(f"Recall: {metrics.box.mr}")
   ```
2. Generate confusion matrix
   ```python
   model.val(data='config/dataset.yaml', plots=True)
   # Saves confusion_matrix.png to runs/detect/val/
   ```
3. Check per-class performance — identify weak classes
4. Visualize false positives and false negatives on 10 test images
5. If mAP < 0.65:
   * Add more training epochs
   * Apply data augmentation (flip, mosaic, HSV shifts)
   * Lower confidence threshold

**Deliverable:** Evaluation report with mAP, precision, recall, confusion matrix ✔

---

### ✅ Phase 5 — Detection Pipeline *(Day 5)*

**Goal:** Run the trained model on yard images and extract structured results.

**Steps:**

1. Write core inference function
   ```python
   # src/detect.py
   from ultralytics import YOLO
   import cv2, json

   def run_detection(image_path, conf_threshold=0.5):
       model = YOLO('models/best.pt')
       results = model.predict(image_path, conf=conf_threshold)
       detections = []
       for box in results[0].boxes:
           detections.append({
               'class': results[0].names[int(box.cls)],
               'confidence': float(box.conf),
               'bbox': box.xyxy[0].tolist()   # [x1, y1, x2, y2]
           })
       return detections
   ```
2. Run on all test images, save raw JSON detections per image
3. Filter low-confidence detections (< 0.5)
4. Group detections by class — count per class per image

**Deliverable:** JSON detection results for all test images ✔

---

### ✅ Phase 6 — OpenCV Annotation & Zone Mapping *(Day 6)*

**Goal:** Draw bounding boxes, define yard zones, map assets to zones.

**Steps:**

**Part A — Bounding Box Annotation**

```python
# src/annotate.py
import cv2

CLASS_COLORS = {
    'truck': (0, 0, 255),           # Red
    'container': (255, 0, 0),       # Blue
    'forklift': (0, 255, 255),      # Yellow
    'person': (0, 255, 0),          # Green
    'safety_vest': (255, 165, 0),   # Orange
    'helmet': (128, 0, 128),        # Purple
}

def annotate_image(image_path, detections):
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        color = CLASS_COLORS.get(det['class'], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img
```

**Part B — Zone Definition**

```json
// config/zones.json
{
  "zones": [
    {"name": "Zone_A_Truck_Bay", "coords": [[0,0],[400,0],[400,300],[0,300]]},
    {"name": "Zone_B_Container_Stack", "coords": [[400,0],[800,0],[800,300],[400,300]]},
    {"name": "Zone_C_Forklift_Lane", "coords": [[0,300],[800,300],[800,600],[0,600]]},
    {"name": "Gate_Entry", "coords": [[350,250],[450,250],[450,350],[350,350]]}
  ]
}
```

**Part C — Zone Mapping**

```python
# src/zone_mapper.py
def point_in_zone(point, zone_coords):
    # Use OpenCV pointPolygonTest
    import cv2, numpy as np
    contour = np.array(zone_coords, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0

def map_detections_to_zones(detections, zones):
    zone_assets = {z['name']: [] for z in zones}
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center = ((x1+x2)//2, (y1+y2)//2)
        for zone in zones:
            if point_in_zone(center, zone['coords']):
                zone_assets[zone['name']].append(det)
    return zone_assets
```

**Deliverable:** Annotated images with zones + asset mapping ✔

---

### ✅ Phase 7 — Use Case Implementation *(Day 7)*

**Goal:** Build the 4 use cases on top of the detection pipeline.

---

#### 🚪 Use Case 1 — Automated Gate Entry Logger

```python
# src/gate_logger.py
import csv, uuid
from datetime import datetime

def log_gate_entry(detections, image_name):
    entries = []
    for det in detections:
        if det['class'] in ['truck', 'container', 'trailer']:
            entries.append({
                'entry_id': str(uuid.uuid4())[:8].upper(),
                'asset_type': det['class'],
                'confidence': round(det['confidence'], 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_source': image_name
            })
    with open('outputs/reports/gate_log.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entries[0].keys())
        writer.writerows(entries)
    return entries
```

---

#### 🚨 Use Case 2 — Anomaly Alert System

```python
# src/anomaly.py
ZONE_RULES = {
    'Zone_A_Truck_Bay': ['truck', 'trailer'],
    'Zone_B_Container_Stack': ['container', 'freight_container'],
    'Zone_C_Forklift_Lane': ['forklift'],
    'Gate_Entry': ['truck', 'car', 'person']
}

def check_anomalies(zone_assets):
    alerts = []
    for zone, assets in zone_assets.items():
        allowed = ZONE_RULES.get(zone, [])
        for asset in assets:
            if asset['class'] not in allowed:
                alerts.append({
                    'alert': f"⚠️ {asset['class']} detected in {zone}",
                    'severity': 'HIGH',
                    'zone': zone,
                    'asset': asset['class']
                })
    return alerts
```

---

#### 📊 Use Case 3 — Occupancy Heatmap

```python
# src/heatmap.py
import numpy as np, cv2

def generate_heatmap(all_detections, image_shape):
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    for det in all_detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        heatmap[y1:y2, x1:x2] += 1
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_colored
```

---

#### ✅ Use Case 4 — Safety Compliance Checker

```python
# src/compliance.py
def check_compliance(detections):
    persons = [d for d in detections if d['class'] == 'person']
    helmets = [d for d in detections if d['class'] == 'helmet']
    vests = [d for d in detections if d['class'] == 'safety_vest']

    total = len(persons)
    compliant = min(len(helmets), len(vests))
    score = (compliant / total * 100) if total > 0 else 100

    return {
        'total_persons': total,
        'compliant_persons': compliant,
        'compliance_score': f"{score:.1f}%",
        'status': '✅ PASS' if score >= 80 else '❌ FAIL'
    }
```

**Deliverable:** All 4 use cases working on test images ✔

---

### ✅ Phase 8 — Report Generation *(Day 7)*

**Goal:** Export structured reports per image.

```python
# src/report.py
import json, csv, os
from datetime import datetime

def generate_report(image_name, detections, zone_assets, alerts, compliance):
    report = {
        'image': image_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_assets_detected': len(detections),
        'asset_count': {},
        'zone_occupancy': {},
        'alerts': alerts,
        'compliance': compliance
    }

    # Asset count per class
    for det in detections:
        cls = det['class']
        report['asset_count'][cls] = report['asset_count'].get(cls, 0) + 1

    # Zone occupancy
    for zone, assets in zone_assets.items():
        report['zone_occupancy'][zone] = {
            'asset_count': len(assets),
            'assets': [a['class'] for a in assets]
        }

    # Save JSON
    json_path = f"outputs/reports/{image_name}_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report
```

**Deliverable:** JSON + CSV reports for every test image ✔

---

### ✅ Phase 9 — Streamlit Dashboard *(Day 8)*

**Goal:** Build an interactive UI that ties everything together.

```python
# src/dashboard.py
import streamlit as st
import cv2, json
from PIL import Image
from detect import run_detection
from annotate import annotate_image
from zone_mapper import map_detections_to_zones
from anomaly import check_anomalies
from compliance import check_compliance
from report import generate_report
from gate_logger import log_gate_entry

st.set_page_config(page_title="SmartYard", page_icon="🚛", layout="wide")
st.title("🚛 SmartYard — AI Yard Asset Detection System")
st.markdown("Upload a yard image to detect assets, map zones, check compliance, and generate a report.")

uploaded = st.file_uploader("Upload Yard Image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    # Save temp image
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded.read())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image("temp_input.jpg")

    with st.spinner("Running detection..."):
        detections = run_detection("temp_input.jpg")
        annotated = annotate_image("temp_input.jpg", detections)
        cv2.imwrite("temp_output.jpg", annotated)

    with col2:
        st.subheader("🔍 Detected Assets")
        st.image("temp_output.jpg")

    st.subheader("📊 Detection Summary")
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Assets", len(detections))

    compliance = check_compliance(detections)
    col4.metric("Compliance Score", compliance['compliance_score'])

    alerts = check_anomalies({})
    col5.metric("Alerts", len(alerts))

    st.subheader("🚨 Anomaly Alerts")
    if alerts:
        for alert in alerts:
            st.error(alert['alert'])
    else:
        st.success("✅ No anomalies detected")

    st.subheader("📁 Download Report")
    report = generate_report(uploaded.name, detections, {}, alerts, compliance)
    st.download_button("Download JSON Report", json.dumps(report, indent=2),
                       file_name="smartyard_report.json", mime="application/json")
```

Run dashboard:

```bash
streamlit run src/dashboard.py
```

**Deliverable:** Working Streamlit dashboard ✔

---

### ✅ Phase 10 — Polish & GitHub *(Day 8)*

**Goal:** Make it presentable for resume and interviews.

**Steps:**

1. Run full pipeline on 10–15 diverse test images
2. Create before/after comparison grid
   ```python
   import matplotlib.pyplot as pltfig, axes = plt.subplots(2, 5, figsize=(20, 8))# Row 1: original, Row 2: annotatedplt.savefig('outputs/comparison_grid.png', dpi=150)
   ```
3. Screenshot the Streamlit dashboard
4. Update GitHub repo:
   * Push all code
   * Add `outputs/` folder with sample annotated images
   * Write clean `README.md` (this file)
5. Add `requirements.txt`
   ```bash
   pip freeze > requirements.txt
   ```
6. Tag release as `v1.0`

**Deliverable:** Public GitHub repo with clean README + screenshots ✔

---

## 🗺️ Full Pipeline Summary

```
Raw Yard Images (from Roboflow datasets)
            ↓
Phase 2: Data Preparation (clean, format, split)
            ↓
Phase 3: Fine-tune YOLOv8s (50 epochs, batch=8, GPU)
            ↓
Phase 4: Evaluate (mAP, confusion matrix, precision/recall)
            ↓
Phase 5: Run Detection (inference on test images)
            ↓
Phase 6: OpenCV Annotation + Zone Mapping
            ↓
Phase 7: Use Case Engine
   ├── 🚪 Gate Entry Logger
   ├── 🚨 Anomaly Alert System
   ├── 📊 Occupancy Heatmap
   └── ✅ Safety Compliance Checker
            ↓
Phase 8: Report Generation (JSON + CSV)
            ↓
Phase 9: Streamlit Dashboard
            ↓
Phase 10: GitHub + Screenshots + Demo
```

---

## 📅 Timeline

| Day      | Phase               | Goal                          |
| -------- | ------------------- | ----------------------------- |
| Day 1    | Setup               | Environment, GPU verified     |
| Day 2    | Data Prep           | Dataset downloaded, formatted |
| Day 3–4 | Training            | best.pt with mAP > 0.70       |
| Day 4    | Evaluation          | Metrics confirmed             |
| Day 5    | Detection           | Inference pipeline working    |
| Day 6    | OpenCV              | Annotated images + zones      |
| Day 7    | Use Cases + Reports | All 4 use cases + JSON/CSV    |
| Day 8    | Dashboard + GitHub  | Streamlit UI + clean repo     |

---

## 📊 Expected Output Per Image

```json
{
  "image": "yard_001.jpg",
  "timestamp": "2026-04-09 10:30:00",
  "total_assets_detected": 11,
  "asset_count": {
    "truck": 3,
    "container": 5,
    "forklift": 2,
    "person": 1
  },
  "zone_occupancy": {
    "Zone_A_Truck_Bay": {"asset_count": 3, "assets": ["truck", "truck", "truck"]},
    "Zone_B_Container_Stack": {"asset_count": 5, "assets": ["container", "container", "container", "container", "container"]},
    "Zone_C_Forklift_Lane": {"asset_count": 2, "assets": ["forklift", "forklift"]}
  },
  "alerts": [
    {"alert": "⚠️ person detected in Zone_B_Container_Stack", "severity": "HIGH"}
  ],
  "compliance": {
    "total_persons": 1,
    "compliant_persons": 0,
    "compliance_score": "0.0%",
    "status": "❌ FAIL"
  }
}
```

---

## 🧰 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
roboflow>=1.1.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/smartyard.git
cd smartyard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (add your Roboflow API key in preprocess.py)
python src/preprocess.py

# 4. Train the model
python src/train.py

# 5. Evaluate the model
python src/evaluate.py

# 6. Run full pipeline on test images
python main.py

# 7. Launch dashboard
streamlit run src/dashboard.py
```

---

## 🏆 Resume Description

> *"Built SmartYard, an end-to-end AI-powered yard asset detection system using YOLOv8 and OpenCV. Fine-tuned on 99K+ logistics images to detect trucks, containers, and forklifts in yard scenes. Implemented zone occupancy mapping, anomaly alerting, automated gate logging, and safety compliance checking. Delivered a Streamlit dashboard with JSON/CSV report export — directly aligned with enterprise Yard Management System functionality."*

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ for supply chain intelligence | Inspired by Kaleris YMS*
