"""
Heatmap Generation Script
Generates occupancy heatmaps from detection data
"""

import numpy as np
import cv2
import os


def generate_heatmap(all_detections, image_shape, normalize=True):
    """
    Generate occupancy heatmap from detections
    
    Args:
        all_detections: List of detection dicts with 'bbox'
        image_shape: Shape of target image (height, width)
        normalize: Whether to normalize heatmap to 0-255
        
    Returns:
        Heatmap image (numpy array)
    """
    # Create empty heatmap
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    # Accumulate detections
    for det in all_detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
        # Add to heatmap
        heatmap[y1:y2, x1:x2] += 1
    
    # Normalize to 0-255
    if normalize and np.max(heatmap) > 0:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    return heatmap


def colorize_heatmap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Convert grayscale heatmap to colored heatmap
    
    Args:
        heatmap: Grayscale heatmap image
        colormap: OpenCV colormap constant
        
    Returns:
        Colored heatmap (numpy array)
    """
    # Convert to uint8 if needed
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    return heatmap_colored


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (numpy array)
        heatmap: Colored heatmap image
        alpha: Transparency factor (0-1)
        
    Returns:
        Blended image with heatmap overlay
    """
    # Ensure same shape
    if image.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def create_heatmap_from_image(image_path, detections, output_path=None, alpha=0.5):
    """
    Create heatmap for a single image
    
    Args:
        image_path: Path to input image
        detections: List of detection dicts
        output_path: Path to save output image
        alpha: Heatmap transparency
        
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Generate heatmap
    heatmap = generate_heatmap(detections, img.shape)
    
    # Colorize
    heatmap_colored = colorize_heatmap(heatmap)
    
    # Overlay
    overlay = overlay_heatmap(img, heatmap_colored, alpha)
    
    # Save output
    if output_path is None:
        os.makedirs('outputs/heatmaps', exist_ok=True)
        output_path = os.path.join('outputs/heatmaps', f"heatmap_{os.path.basename(image_path)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Heatmap saved: {output_path}")
    
    return heatmap, overlay


def create_aggregated_heatmap(all_detections_dict, reference_image, output_path='outputs/heatmaps/aggregated_heatmap.png'):
    """
    Create aggregated heatmap from multiple images
    
    Args:
        all_detections_dict: Dict mapping image names to detections
        reference_image: Path to reference image for dimensions
        output_path: Path to save output
        
    Returns:
        Aggregated heatmap image
    """
    # Get image dimensions
    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    
    # Aggregate all detections
    all_dets = []
    for image_name, detections in all_detections_dict.items():
        all_dets.extend(detections)
    
    # Generate heatmap
    heatmap = generate_heatmap(all_dets, img.shape)
    heatmap_colored = colorize_heatmap(heatmap)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, heatmap_colored)
    print(f"Aggregated heatmap saved: {output_path}")
    
    return heatmap_colored


def main():
    """Test heatmap generation"""
    print("SmartYard Heatmap Generator")
    print("="*50)
    
    # Sample detections
    sample_detections = [
        {'bbox': [100, 100, 300, 250]},
        {'bbox': [150, 120, 350, 270]},
        {'bbox': [400, 300, 550, 450]},
    ]
    
    print("\nTo generate heatmaps:")
    print("  from heatmap import create_heatmap_from_image")
    print("  heatmap, overlay = create_heatmap_from_image('image.jpg', detections)")
    print("\nOr for aggregated heatmaps:")
    print("  from heatmap import create_aggregated_heatmap")
    print("  heatmap = create_aggregated_heatmap(all_detections, 'reference.jpg')")


if __name__ == "__main__":
    main()
