"""
Image Annotation Script
Draws bounding boxes and labels on detected objects
"""

import cv2
import os


# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    'truck': (0, 0, 255),           # Red
    'trailer': (128, 0, 128),       # Purple
    'container': (255, 0, 0),       # Blue
    'freight_container': (255, 0, 0),  # Blue (same as container)
    'forklift': (0, 255, 255),      # Yellow
    'person': (0, 255, 0),          # Green
    'helmet': (255, 255, 0),        # Cyan
    'safety_vest': (0, 165, 255),   # Orange
    'car': (200, 200, 200),         # Gray
}


def annotate_image(image_path, detections, output_path=None, show_labels=True):
    """
    Annotate image with bounding boxes and labels
    
    Args:
        image_path: Path to input image
        detections: List of detection dicts with 'class', 'confidence', 'bbox'
        output_path: Path to save annotated image (default: outputs/annotated/)
        show_labels: Whether to show class labels
        
    Returns:
        Annotated image (numpy array)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        color = CLASS_COLORS.get(det['class'], (255, 255, 255))  # White default
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        if show_labels:
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(
                img,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
    
    # Save annotated image
    if output_path is None:
        os.makedirs('outputs/annotated', exist_ok=True)
        output_path = os.path.join('outputs/annotated', os.path.basename(image_path))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved: {output_path}")
    
    return img


def annotate_batch(image_detections, output_dir='outputs/annotated'):
    """
    Anulate multiple images
    
    Args:
        image_detections: Dict mapping image paths to detections
        output_dir: Directory to save annotated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path, detections in image_detections.items():
        try:
            annotate_image(image_path, detections, output_dir)
        except Exception as e:
            print(f"Error annotating {image_path}: {e}")


def main():
    """Test annotation with sample data"""
    # Example usage
    sample_detections = [
        {
            'class': 'truck',
            'confidence': 0.92,
            'bbox': [100, 100, 300, 250]
        },
        {
            'class': 'person',
            'confidence': 0.85,
            'bbox': [350, 200, 400, 350]
        },
        {
            'class': 'forklift',
            'confidence': 0.78,
            'bbox': [450, 300, 550, 400]
        }
    ]
    
    print("SmartYard Image Annotation")
    print("="*50)
    print("To annotate images, use:")
    print("  from annotate import annotate_image")
    print("  annotate_image('path/to/image.jpg', detections)")
    print("\nOr run batch annotation:")
    print("  from annotate import annotate_batch")
    print("  annotate_batch(image_detections_dict)")


if __name__ == "__main__":
    main()
