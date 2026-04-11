"""
SmartYard - Main Entry Point
Runs the full detection pipeline on test images
"""

import os
import json
from pathlib import Path

# Import SmartYard modules
from src.core.detect import run_detection, run_batch_detection
from src.analytics.annotate import annotate_image
from src.analytics.zone_mapper import load_zones, map_detections_to_zones, calculate_zone_occupancy
from src.analytics.anomaly import check_anomalies
from src.analytics.compliance import check_compliance
from src.analytics.gate_logger import log_gate_entry
from src.analytics.report import generate_report, generate_batch_reports
from src.analytics.heatmap import create_heatmap_from_image


def run_pipeline_single(image_path, output_dir='outputs'):
    """
    Run full SmartYard pipeline on a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    image_name = os.path.basename(image_path)
    
    # Step 1: Run detection
    print("\n[Step 1] Running object detection...")
    detections = run_detection(image_path)
    print(f"   Detected {len(detections)} objects")
    
    # Step 2: Annotate image
    print("\n[Step 2] Annotating image...")
    os.makedirs(f'{output_dir}/annotated', exist_ok=True)
    output_path = f"{output_dir}/annotated/{image_name}"
    annotated = annotate_image(image_path, detections, output_path)
    
    # Step 3: Load zones and map detections
    print("\n[Step 3] Mapping assets to zones...")
    zones = load_zones()
    zone_assets = map_detections_to_zones(detections, zones)
    zone_occupancy = calculate_zone_occupancy(zone_assets, zones)
    
    active_zones = {k: v for k, v in zone_occupancy.items() if v['asset_count'] > 0}
    print(f"   Assets mapped to {len(active_zones)} zones")
    
    # Step 4: Check anomalies
    print("\n[Step 4] Checking for anomalies...")
    alerts = check_anomalies(zone_assets)
    print(f"   Found {len(alerts)} anomalies")
    
    # Step 5: Check compliance
    print("\n[Step 5] Checking safety compliance...")
    compliance = check_compliance(detections)
    print(f"   Compliance: {compliance['status']} ({compliance['compliance_score']})")
    
    # Step 6: Log gate entries
    print("\n[Step 6] Logging gate entries...")
    gate_entries = log_gate_entry(detections, image_name)
    print(f"   Logged {len(gate_entries)} entries")
    
    # Step 7: Generate report
    print("\n[Step 7] Generating report...")
    report = generate_report(
        image_name=image_name,
        detections=detections,
        zone_occupancy=zone_occupancy,
        alerts=alerts,
        compliance=compliance,
        gate_entries=gate_entries
    )
    
    # Step 8: Generate heatmap (optional)
    print("\n[Step 8] Generating heatmap...")
    try:
        create_heatmap_from_image(image_path, detections)
    except Exception as e:
        print(f"   Heatmap generation skipped: {e}")
    
    print(f"\nPipeline complete for {image_name}")
    print(f"   Results saved to: {output_dir}")
    
    return {
        'image': image_name,
        'detections': detections,
        'zone_occupancy': zone_occupancy,
        'alerts': alerts,
        'compliance': compliance,
        'gate_entries': gate_entries,
        'report': report
    }


def run_pipeline_batch(image_dir, output_dir='outputs'):
    """
    Run full pipeline on all images in a directory
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save outputs
    """
    print("="*60)
    print("SmartYard Batch Processing Pipeline")
    print("="*60)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(ext))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nFound {len(image_paths)} images to process")
    
    all_results = {}
    all_detections = {}
    all_zone_occupancy = {}
    all_alerts = {}
    all_compliance = {}
    
    for img_path in image_paths:
        result = run_pipeline_single(str(img_path), output_dir)
        all_results[img_path.name] = result
        all_detections[img_path.name] = result['detections']
        all_zone_occupancy[img_path.name] = result['zone_occupancy']
        all_alerts[img_path.name] = result['alerts']
        all_compliance[img_path.name] = result['compliance']
    
    # Generate batch reports
    print("\n" + "="*60)
    print("Generating batch reports...")
    print("="*60)
    generate_batch_reports(
        all_detections,
        all_zone_occupancy,
        all_alerts,
        all_compliance
    )
    
    # Summary
    print("\n" + "="*60)
    print("Batch Processing Summary")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    
    total_assets = sum(r['report']['total_assets_detected'] for r in all_results.values())
    total_alerts = sum(len(r['alerts']) for r in all_results.values())
    
    print(f"Total assets detected: {total_assets}")
    print(f"Total alerts generated: {total_alerts}")
    
    # Asset breakdown
    asset_counts = {}
    for result in all_results.values():
        for cls, count in result['report']['asset_count'].items():
            asset_counts[cls] = asset_counts.get(cls, 0) + count
    
    print("\nAsset breakdown:")
    for cls, count in sorted(asset_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:25s}: {count}")
    
    return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SmartYard Full Pipeline')
    parser.add_argument(
        '--input',
        type=str,
        default='data/test_images',
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best.pt',
        help='Path to model weights'
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        print("\nPlease add test images to data/test_images/ or specify a valid path")
        return
    
    # Run pipeline
    if os.path.isfile(args.input):
        # Single image
        run_pipeline_single(args.input, args.output)
    else:
        # Directory
        run_pipeline_batch(args.input, args.output)


if __name__ == "__main__":
    main()
