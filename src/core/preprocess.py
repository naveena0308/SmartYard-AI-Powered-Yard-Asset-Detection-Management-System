"""
Dataset Download and Preparation Script
Downloads datasets from Roboflow, filters classes, merges, and formats for YOLOv8
"""

import os
import shutil
import yaml
import cv2
import random
from pathlib import Path
from roboflow import Roboflow
from tqdm import tqdm

# Configuration
ROBOFLOW_API_KEY = "eEP7MOHEiNm1klR038XO"
TARGET_CLASSES = ['truck', 'trailer', 'container', 'forklift', 'person', 'helmet', 'safety_vest', 'car']
IMAGE_SIZE = 640
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Dataset configurations
DATASET_CONFIGS = [
    {
        "workspace": "large-benchmark-datasets",
        "project": "logistics-sz9jr",
        "version": 1,
        "mapping": {
            "Truck": "truck",
            "Car": "car",
            "Freight Container": "container",
            "Forklift": "forklift",
            "Helmet": "helmet",
            "Safety Vest": "safety_vest",
            "Person": "person",
            "Van": "car"
        }
    },
    {
        "workspace": "roboflow-universe-projects",
        "project": "yard-management-system",
        "version": 1,
        "mapping": {
            "Container": "container",
            "Chassis ID": "trailer"
        }
    },
    {
        "workspace": "data-woimc",
        "project": "truck-container-q4wrp",
        "version": 3,
        "mapping": {
            "Truck": "truck",
            "Container": "container"
        }
    }
]

def download_dataset(workspace, project_name, version):
    """Download dataset from Roboflow"""
    print(f"\n[INFO] Downloading {project_name} (v{version})...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")
    return Path(dataset.location)

def get_class_remapping(dataset_path, mapping):
    """Create a map from source class index to target class index"""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    source_names = data.get('names', [])
    index_map = {}
    
    for i, name in enumerate(source_names):
        name_lower = name.lower() if isinstance(name, str) else name
        for map_key, target_class in mapping.items():
            if map_key.lower() == name_lower:
                target_idx = TARGET_CLASSES.index(target_class)
                index_map[i] = target_idx
                break
            
    return index_map

def process_and_collect(dataset_path, index_map):
    """Filter labels and collect valid image/label pairs"""
    valid_pairs = []
    
    # Roboflow usually exports in train/test/valid folders
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        
        if not img_dir.exists(): continue
        
        print(f"  Processing {split} split of {dataset_path.name}...")
        for img_path in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists(): continue
            
            # Filter and remap labels
            new_labels = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    class_id = int(parts[0])
                    if class_id in index_map:
                        parts[0] = str(index_map[class_id])
                        new_labels.append(" ".join(parts))
            
            if new_labels:
                # Save temp cleaned label
                temp_lbl_path = lbl_path.parent / f"{img_path.stem}_cleaned.txt"
                with open(temp_lbl_path, 'w') as f:
                    f.write("\n".join(new_labels))
                valid_pairs.append((img_path, temp_lbl_path))
                
    return valid_pairs

def prepare_directories():
    """Create necessary directories"""
    base = Path("data/processed")
    for split in ['train', 'val', 'test']:
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
    return base

def finalize_dataset(all_pairs, output_base):
    """Shuffle, split, and move files to processed directory"""
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)
    
    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    for split_name, pairs in splits.items():
        print(f"\n[INFO] Finalizing {split_name} split ({len(pairs)} images)...")
        img_out = output_base / split_name / "images"
        lbl_out = output_base / split_name / "labels"
        
        for img_path, lbl_path in tqdm(pairs):
            # Resize and save image
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                cv2.imwrite(str(img_out / img_path.name), img)
                
                # Move cleaned label
                shutil.copy(lbl_path, lbl_out / f"{img_path.stem}.txt")

def main():
    print("="*50)
    print("SmartYard Dataset Preparation Pipeline")
    print("="*50)
    
    output_base = prepare_directories()
    all_valid_pairs = []
    
    for cfg in DATASET_CONFIGS:
        try:
            loc = download_dataset(cfg["workspace"], cfg["project"], cfg["version"])
            idx_map = get_class_remapping(loc, cfg["mapping"])
            pairs = process_and_collect(loc, idx_map)
            all_valid_pairs.extend(pairs)
        except Exception as e:
            print(f"[ERROR] Failed to process {cfg['project']}: {e}")
            
    if all_valid_pairs:
        finalize_dataset(all_valid_pairs, output_base)
        print(f"\n✅ Dataset preparation complete! Total images: {len(all_valid_pairs)}")
    else:
        print("\n❌ No valid data pairs found.")

if __name__ == "__main__":
    main()
