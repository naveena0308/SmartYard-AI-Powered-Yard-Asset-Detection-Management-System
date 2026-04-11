"""
Zone Mapping Script
Defines yard zones and maps detections to zones
"""

import cv2
import numpy as np
import json
import os


def load_zones(config_path='config/zones.json'):
    """
    Load zone definitions from JSON config
    
    Args:
        config_path: Path to zones.json config file
        
    Returns:
        List of zone dictionaries
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['zones']


def point_in_zone(point, zone_coords):
    """
    Check if a point is inside a polygon zone
    
    Args:
        point: (x, y) tuple
        zone_coords: List of [x, y] coordinates defining the polygon
        
    Returns:
        True if point is inside zone, False otherwise
    """
    contour = np.array(zone_coords, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def map_detections_to_zones(detections, zones):
    """
    Map detected assets to their respective zones
    
    Args:
        detections: List of detection dicts with 'class', 'confidence', 'bbox'
        zones: List of zone dicts with 'name' and 'coords'
        
    Returns:
        Dict mapping zone names to lists of detections in that zone
    """
    zone_assets = {z['name']: [] for z in zones}
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Calculate center point of bounding box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Check which zone contains the center
        for zone in zones:
            if point_in_zone(center, zone['coords']):
                zone_assets[zone['name']].append(det)
                break  # Asset mapped to first matching zone
    
    return zone_assets


def calculate_zone_occupancy(zone_assets, zones):
    """
    Calculate occupancy statistics for each zone
    
    Args:
        zone_assets: Dict mapping zone names to detections
        zones: List of zone definitions
        
    Returns:
        Dict with zone occupancy info
    """
    occupancy = {}
    
    for zone in zones:
        zone_name = zone['name']
        assets = zone_assets.get(zone_name, [])
        
        # Count assets by class
        asset_counts = {}
        for asset in assets:
            cls = asset['class']
            asset_counts[cls] = asset_counts.get(cls, 0) + 1
        
        occupancy[zone_name] = {
            'asset_count': len(assets),
            'assets': [a['class'] for a in assets],
            'asset_counts': asset_counts
        }
    
    return occupancy


def visualize_zones(image_path, zones, zone_assets, output_path=None):
    """
    Draw zone boundaries on image
    
    Args:
        image_path: Path to input image
        zones: List of zone definitions
        zone_assets: Dict mapping zone names to detections
        output_path: Path to save output image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Zone colors (BGR)
    zone_colors = [
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    # Draw zones
    for i, zone in enumerate(zones):
        color = zone_colors[i % len(zone_colors)]
        coords = np.array(zone['coords'], dtype=np.int32)
        
        # Draw polygon
        cv2.polylines(img, [coords], True, color, 2)
        
        # Add zone label
        asset_count = len(zone_assets.get(zone['name'], []))
        label = f"{zone['name']} ({asset_count})"
        cv2.putText(
            img,
            label,
            (coords[0][0], coords[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    # Save output
    if output_path is None:
        os.makedirs('outputs/annotated', exist_ok=True)
        output_path = os.path.join('outputs/annotated', f"zones_{os.path.basename(image_path)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Zone visualization saved: {output_path}")
    
    return img


def main():
    """Test zone mapping"""
    print("SmartYard Zone Mapping")
    print("="*50)
    
    # Load zones
    zones = load_zones()
    print(f"Loaded {len(zones)} zones:")
    for zone in zones:
        print(f"  - {zone['name']}")
    
    print("\nTo use zone mapping:")
    print("  from zone_mapper import load_zones, map_detections_to_zones")
    print("  zones = load_zones()")
    print("  zone_assets = map_detections_to_zones(detections, zones)")


if __name__ == "__main__":
    main()
