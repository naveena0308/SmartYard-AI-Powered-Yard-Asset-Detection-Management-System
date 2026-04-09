"""
Model Evaluation Script
Evaluates trained YOLOv8 model and generates metrics
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def evaluate_model(
    model_path='models/best.pt',
    data_config='config/dataset.yaml',
    split='test',
    conf_threshold=0.5
):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model weights
        data_config: Path to dataset YAML config
        split: Data split to evaluate (train/val/test)
        conf_threshold: Confidence threshold for detections
    """
    print("="*50)
    print("SmartYard Model Evaluation")
    print("="*50)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print(f"\nEvaluating on {split} set...")
    metrics = model.val(
        data=data_config,
        split=split,
        conf=conf_threshold,
        save=True,
        plots=True
    )
    
    # Print metrics
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class performance
    print("\nPer-Class Performance:")
    print("-" * 50)
    for i, name in enumerate(model.names.values()):
        print(f"  {name:20s} - mAP@50: {metrics.box.maps50[i]:.4f}")
    
    # Save metrics to JSON
    metrics_dict = {
        'mAP@50': float(metrics.box.map50),
        'mAP@50-95': float(metrics.box.map),
        'Precision': float(metrics.box.mp),
        'Recall': float(metrics.box.mr),
        'per_class_mAP50': {
            name: float(metrics.box.maps50[i])
            for i, name in enumerate(model.names.values())
        }
    }
    
    os.makedirs('outputs/reports', exist_ok=True)
    with open('outputs/reports/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\n✅ Metrics saved to: outputs/reports/evaluation_metrics.json")
    print(f"📊 Confusion matrix and plots saved to: runs/detect/val/")
    
    return metrics


def visualize_results():
    """Visualize training results and confusion matrix"""
    print("\nGenerating visualizations...")
    
    # Check if confusion matrix exists
    cm_path = 'runs/detect/val/confusion_matrix.png'
    if os.path.exists(cm_path):
        print(f"Confusion matrix available at: {cm_path}")
    
    # You can add more visualization code here
    # e.g., plot PR curves, F1 curves, etc.


def main():
    """Main evaluation function"""
    # Evaluate model
    metrics = evaluate_model(
        model_path='models/best.pt',
        data_config='config/dataset.yaml',
        split='test',
        conf_threshold=0.5
    )
    
    # Visualize results
    visualize_results()
    
    # Check if model meets threshold
    if metrics.box.map50 < 0.65:
        print("\n⚠️  Warning: mAP@50 < 0.65")
        print("Consider:")
        print("  - Increasing training epochs")
        print("  - Adding data augmentation")
        print("  - Lowering confidence threshold")
    else:
        print(f"\n✅ Model meets performance threshold (mAP@50 >= 0.65)")


if __name__ == "__main__":
    main()
