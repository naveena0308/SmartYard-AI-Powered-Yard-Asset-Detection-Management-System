"""
Report Generation Script
Generates structured JSON and CSV reports
"""

import json
import csv
import os
from datetime import datetime


def generate_report(
    image_name,
    detections,
    zone_occupancy=None,
    alerts=None,
    compliance=None,
    gate_entries=None,
    output_dir='outputs/reports'
):
    """
    Generate comprehensive report for an image
    
    Args:
        image_name: Name of the source image
        detections: List of detection dicts
        zone_occupancy: Dict mapping zones to occupancy info
        alerts: List of alert dicts
        compliance: Compliance result dict
        gate_entries: List of gate entry dicts
        output_dir: Directory to save reports
        
    Returns:
        Report dictionary
    """
    # Build report
    report = {
        'image': image_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_assets_detected': len(detections),
        'asset_count': {},
        'zone_occupancy': zone_occupancy or {},
        'alerts': alerts or [],
        'compliance': compliance or {},
        'gate_entries': gate_entries or []
    }
    
    # Count assets by class
    for det in detections:
        cls = det['class']
        report['asset_count'][cls] = report['asset_count'].get(cls, 0) + 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved: {json_path}")
    
    return report


def generate_csv_report(detections, output_file='outputs/reports/detections_summary.csv'):
    """
    Generate CSV summary of all detections
    
    Args:
        detections: List of detection dicts
        output_file: Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        fieldnames = ['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for det in detections:
            writer.writerow({
                'image': det.get('image', 'unknown'),
                'class': det['class'],
                'confidence': det['confidence'],
                'x1': det['bbox'][0],
                'y1': det['bbox'][1],
                'x2': det['bbox'][2],
                'y2': det['bbox'][3],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    print(f"CSV report saved: {output_file}")


def generate_batch_reports(
    image_detections,
    zone_occupancy_map=None,
    alerts_map=None,
    compliance_map=None,
    output_dir='outputs/reports'
):
    """
    Generate reports for multiple images
    
    Args:
        image_detections: Dict mapping image names to detections
        zone_occupancy_map: Dict mapping image names to zone occupancy
        alerts_map: Dict mapping image names to alerts
        compliance_map: Dict mapping image names to compliance results
        output_dir: Directory to save reports
    """
    reports = []
    
    for image_name, detections in image_detections.items():
        report = generate_report(
            image_name=image_name,
            detections=detections,
            zone_occupancy=zone_occupancy_map.get(image_name) if zone_occupancy_map else None,
            alerts=alerts_map.get(image_name) if alerts_map else None,
            compliance=compliance_map.get(image_name) if compliance_map else None,
            output_dir=output_dir
        )
        reports.append(report)
    
    # Generate summary CSV
    all_detections = []
    for image_name, detections in image_detections.items():
        for det in detections:
            det_copy = det.copy()
            det_copy['image'] = image_name
            all_detections.append(det_copy)
    
    generate_csv_report(all_detections)
    
    print(f"\n✅ Generated {len(reports)} reports")
    return reports


def read_report(report_path):
    """
    Read a JSON report
    
    Args:
        report_path: Path to JSON report file
        
    Returns:
        Report dictionary
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    return report


def generate_summary_report(all_reports, output_file='outputs/reports/summary_report.json'):
    """
    Generate summary report across all images
    
    Args:
        all_reports: List of report dicts
        output_file: Path to save summary report
    """
    summary = {
        'total_images': len(all_reports),
        'total_assets': 0,
        'asset_counts': {},
        'total_alerts': 0,
        'images_with_alerts': 0,
        'average_compliance_score': 0.0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    compliance_scores = []
    
    for report in all_reports:
        summary['total_assets'] += report['total_assets_detected']
        summary['total_alerts'] += len(report.get('alerts', []))
        
        if report.get('alerts'):
            summary['images_with_alerts'] += 1
        
        # Aggregate asset counts
        for cls, count in report.get('asset_count', {}).items():
            summary['asset_counts'][cls] = summary['asset_counts'].get(cls, 0) + count
        
        # Collect compliance scores
        if report.get('compliance') and report['compliance'].get('compliance_score'):
            score_str = report['compliance']['compliance_score'].replace('%', '')
            try:
                compliance_scores.append(float(score_str))
            except ValueError:
                pass
    
    # Calculate average compliance
    if compliance_scores:
        summary['average_compliance_score'] = f"{sum(compliance_scores) / len(compliance_scores):.1f}%"
    
    # Save summary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report saved: {output_file}")
    return summary


def main():
    """Test report generation"""
    print("SmartYard Report Generator")
    print("="*50)
    
    # Sample data
    sample_detections = [
        {'class': 'truck', 'confidence': 0.92, 'bbox': [100, 100, 300, 250]},
        {'class': 'person', 'confidence': 0.85, 'bbox': [350, 200, 400, 350]},
    ]
    
    report = generate_report(
        image_name="test_image.jpg",
        detections=sample_detections,
        zone_occupancy={'Zone_A': {'asset_count': 2}},
        alerts=[],
        compliance={'status': '✅ PASS', 'compliance_score': '100.0%'}
    )
    
    print(f"\nGenerated report for: {report['image']}")
    print(f"Total assets: {report['total_assets_detected']}")
    
    print("\nTo use report generator:")
    print("  from report import generate_report")
    print("  report = generate_report('image.jpg', detections, zone_occupancy, alerts, compliance)")


if __name__ == "__main__":
    main()
