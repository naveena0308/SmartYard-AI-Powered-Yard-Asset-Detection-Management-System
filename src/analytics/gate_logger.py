"""
Gate Entry Logger
Automated logging of vehicles at gate entry points
"""

import csv
import uuid
import os
from datetime import datetime


def log_gate_entry(detections, image_name="unknown", output_file='outputs/reports/gate_log.csv'):
    """
    Log gate entry for detected vehicles
    
    Args:
        detections: List of detection dicts
        image_name: Name of source image
        output_file: Path to CSV output file
        
    Returns:
        List of entry dicts that were logged
    """
    # Vehicle classes to log
    vehicle_classes = ['truck', 'container', 'trailer', 'car']
    
    # Filter for vehicles
    entries = []
    for det in detections:
        if det['class'] in vehicle_classes:
            entries.append({
                'entry_id': str(uuid.uuid4())[:8].upper(),
                'asset_type': det['class'],
                'confidence': round(det['confidence'], 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_source': image_name,
                'bbox': str(det.get('bbox', []))
            })
    
    if not entries:
        return []
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to CSV
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        fieldnames = ['entry_id', 'asset_type', 'confidence', 'timestamp', 'image_source', 'bbox']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(entries)
    
    print(f"Logged {len(entries)} gate entries to {output_file}")
    return entries


def log_batch_gate_entry(image_detections, output_file='outputs/reports/gate_log.csv'):
    """
    Log gate entries for multiple images
    
    Args:
        image_detections: Dict mapping image names to detections
        output_file: Path to CSV output file
    """
    total_entries = 0
    
    for image_name, detections in image_detections.items():
        entries = log_gate_entry(detections, image_name, output_file)
        total_entries += len(entries)
    
    print(f"\n✅ Total gate entries logged: {total_entries}")


def read_gate_log(output_file='outputs/reports/gate_log.csv'):
    """
    Read and display gate log
    
    Args:
        output_file: Path to gate log CSV
        
    Returns:
        List of entry dicts
    """
    if not os.path.exists(output_file):
        print(f"Gate log not found: {output_file}")
        return []
    
    entries = []
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    
    print(f"Gate log contains {len(entries)} entries")
    return entries


def generate_gate_summary(output_file='outputs/reports/gate_log.csv'):
    """
    Generate summary statistics from gate log
    
    Args:
        output_file: Path to gate log CSV
    """
    entries = read_gate_log(output_file)
    
    if not entries:
        return
    
    # Count by asset type
    asset_counts = {}
    for entry in entries:
        asset_type = entry['asset_type']
        asset_counts[asset_type] = asset_counts.get(asset_type, 0) + 1
    
    print("\n📊 GATE ENTRY SUMMARY")
    print("=" * 50)
    print(f"Total Entries: {len(entries)}")
    print("\nBy Asset Type:")
    for asset_type, count in asset_counts.items():
        print(f"  {asset_type:20s}: {count}")


def main():
    """Test gate logger"""
    print("SmartYard Gate Entry Logger")
    print("="*50)
    
    # Sample detections
    sample_detections = [
        {'class': 'truck', 'confidence': 0.92, 'bbox': [100, 100, 300, 250]},
        {'class': 'container', 'confidence': 0.85, 'bbox': [350, 200, 500, 400]},
        {'class': 'person', 'confidence': 0.80, 'bbox': [50, 50, 100, 150]}  # Not a vehicle
    ]
    
    entries = log_gate_entry(sample_detections, "test_image.jpg")
    
    print(f"\nLogged entries:")
    for entry in entries:
        print(f"  {entry['entry_id']} - {entry['asset_type']} ({entry['confidence']})")
    
    print("\nTo use gate logger:")
    print("  from gate_logger import log_gate_entry")
    print("  entries = log_gate_entry(detections, 'image_name.jpg')")


if __name__ == "__main__":
    main()
