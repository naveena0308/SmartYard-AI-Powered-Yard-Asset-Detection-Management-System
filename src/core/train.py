"""
Model Training Script
Fine-tunes YOLOv8 on logistics/yard dataset
"""

from ultralytics import YOLO
import torch
import os


def check_gpu():
    """Check GPU availability"""
    print("Checking GPU availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected. Training will use CPU (slow).")


def train_model(
    model_path='yolov8s.pt',
    data_config='config/dataset.yaml',
    epochs=50,
    img_size=640,
    batch_size=8,
    device=0,
    patience=10,
    project='models',
    name='smartyard_v1'
):
    """
    Train/fine-tune YOLOv8 model
    
    Args:
        model_path: Path to pretrained weights or model size
        data_config: Path to dataset YAML config
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size (reduce if OOM)
        device: GPU device (0) or CPU ('cpu')
        patience: Early stopping patience
        project: Project directory for saving weights
        name: Run name
    """
    print("="*50)
    print("SmartYard Model Training")
    print("="*50)
    
    # Check GPU
    check_gpu()
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Image size: {img_size}")
    
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        save=True,
        project=project,
        name=name,
        # Optional augmentations
        # augment=True,  # Enable if overfitting
        # hsv_h=0.015,   # HSV-Hue augmentation
        # hsv_s=0.7,     # HSV-Saturation augmentation
        # hsv_v=0.4,     # HSV-Value augmentation
        # flipud=0.0,    # Vertical flip
        # fliplr=0.5,    # Horizontal flip
    )
    
    print("\n✅ Training complete!")
    print(f"Best weights saved to: {project}/{name}/weights/best.pt")
    print(f"Last weights saved to: {project}/{name}/weights/last.pt")
    
    # Copy best weights to models/best.pt for easy access
    best_pt = f"{project}/{name}/weights/best.pt"
    if os.path.exists(best_pt):
        os.makedirs('models', exist_ok=True)
        import shutil
        shutil.copy(best_pt, 'models/best.pt')
        print(f"Copied best weights to: models/best.pt")
    
    return results


def main():
    """Main training function"""
    # Training configuration
    train_model(
        model_path='yolov8s.pt',  # Or load Roboflow pretrained weights
        data_config='config/dataset.yaml',
        epochs=50,
        img_size=640,
        batch_size=8,  # Reduce to 4 if GPU runs out of memory
        device=0,      # Use GPU (set to 'cpu' if no GPU)
        patience=10,
        project='models',
        name='smartyard_v1'
    )


if __name__ == "__main__":
    main()
