"""
Detection Pipeline Script
Runs inference on yard images and extracts structured results
"""

from ultralytics import YOLO
import json
import os
from pathlib import Path


def run_detection(image_path, model_path='models/best.pt', conf_threshold=0.5):
    """
    Run object detection on a single image
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detection dictionaries with class, confidence, and bbox
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(image_path, conf=conf_threshold)
    
    # Extract detections
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': results[0].names[int(box.cls)],
            'confidence': round(float(box.conf), 4),
            'bbox': [round(float(x), 2) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
        })
    
    return detections


def run_batch_detection(image_dir, model_path='models/best.pt', conf_threshold=0.5, output_dir='outputs/reports'):
    """
    Run detection on all images in a directory
    
    Args:
        image_dir: Directory containing images to process
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        output_dir: Directory to save detection results
        
    Returns:
        Dictionary mapping image names to detections
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(ext))
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    all_detections = {}
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        detections = run_detection(str(img_path), model_path, conf_threshold)
        
        # Filter low-confidence detections
        detections = [d for d in detections if d['confidence'] >= conf_threshold]
        
        all_detections[img_path.name] = detections
        
        # Group by class
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"  Detected {len(detections)} objects:")
        for cls, count in class_counts.items():
            print(f"    - {cls}: {count}")
        
        # Save JSON
        output_path = os.path.join(output_dir, f"{img_path.stem}_detections.json")
        with open(output_path, 'w') as f:
            json.dump({
                'image': img_path.name,
                'total_detections': len(detections),
                'detections': detections,
                'class_counts': class_counts
            }, f, indent=2)
    
    print(f"\n✅ Detection complete! Results saved to {output_dir}")
    return all_detections


def main():
    """Main detection pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run object detection on yard images')
    parser.add_argument('--input', type=str, default='data/test_images',
                       help='Input image or directory')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs/reports',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*50)
    print("SmartYard Detection Pipeline")
    print("="*50)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image
        detections = run_detection(args.input, args.model, args.conf)
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2f}")
    else:
        # Directory
        run_batch_detection(args.input, args.model, args.conf, args.output)


if __name__ == "__main__":
    main()
