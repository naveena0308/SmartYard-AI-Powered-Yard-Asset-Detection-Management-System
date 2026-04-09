"""
Dataset Download and Preparation Script
Downloads datasets from Roboflow, filters classes, merges, and formats for YOLOv8
"""

import os
import shutil
from pathlib import Path
from roboflow import Roboflow
import cv2
import glob


# Configuration
ROBOFLOW_API_KEY = "YOUR_API_KEY"  # Replace with your API key
TARGET_CLASSES = [
    'truck', 'trailer', 'container', 'freight_container', 
    'forklift', 'person', 'helmet', 'safety_vest'
]
IMAGE_SIZE = 640
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1


def download_dataset(workspace, project_name, version=1):
    """Download dataset from Roboflow"""
    print(f"Downloading dataset: {project_name}")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")
    print(f"Downloaded to: {dataset.location}")
    return dataset


def resize_image(image_path, target_size=640):
    """Resize image to target size"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    resized = cv2.resize(img, (target_size, target_size))
    cv2.imwrite(image_path, resized)
    return True


def prepare_directories():
    """Create necessary directories"""
    dirs = [
        'data/processed/train/images',
        'data/processed/train/labels',
        'data/processed/val/images',
        'data/processed/val/labels',
        'data/processed/test/images',
        'data/processed/test/labels',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def merge_and_split_datasets(dataset_paths):
    """
    Merge multiple datasets and split into train/val/test
    TODO: Implement dataset merging logic based on your specific needs
    """
    print("Merging and splitting datasets...")
    # This is a placeholder - implement based on your dataset structure
    print("Dataset merging not yet implemented")
    pass


def main():
    """Main preprocessing pipeline"""
    print("="*50)
    print("SmartYard Dataset Preparation")
    print("="*50)
    
    # Create directories
    prepare_directories()
    
    # Download datasets
    print("\nDownloading datasets from Roboflow...")
    
    # Dataset 1: Logistics Object Detection
    # dataset1 = download_dataset(
    #     workspace="large-benchmark-datasets",
    #     project_name="logistics-sz9jr",
    #     version=1
    # )
    
    # Dataset 2: Yard Management System
    # dataset2 = download_dataset(
    #     workspace="roboflow-universe-projects",
    #     project_name="yard-management-system",
    #     version=1
    # )
    
    # Dataset 3: Truck Container
    # dataset3 = download_dataset(
    #     workspace="data-woimc",
    #     project_name="truck-container-q4wrp",
    #     version=3
    # )
    
    # Merge and split datasets
    # merge_and_split_datasets([dataset1, dataset2, dataset3])
    
    print("\n✅ Dataset preparation complete!")
    print("⚠️  Remember to:")
    print("   1. Add your Roboflow API key")
    print("   2. Uncomment dataset download calls")
    print("   3. Implement merge_and_split_datasets() for your specific data")


if __name__ == "__main__":
    main()
