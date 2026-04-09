"""
Anomaly Alert System
Detects zone violations and unauthorized assets
"""

import json


# Zone rules: define allowed asset classes per zone
ZONE_RULES = {
    'Zone_A_Truck_Bay': ['truck', 'trailer'],
    'Zone_B_Container_Stack': ['container', 'freight_container'],
    'Zone_C_Forklift_Lane': ['forklift'],
    'Gate_Entry': ['truck', 'car', 'person']
}


def check_anomalies(zone_assets, zone_rules=None):
    """
    Check for zone violations and anomalies
    
    Args:
        zone_assets: Dict mapping zone names to lists of detections
        zone_rules: Dict mapping zone names to allowed classes (optional)
        
    Returns:
        List of alert dicts with 'alert', 'severity', 'zone', 'asset'
    """
    if zone_rules is None:
        zone_rules = ZONE_RULES
    
    alerts = []
    
    for zone, assets in zone_assets.items():
        allowed = zone_rules.get(zone, [])
        
        # Skip zones with no rules defined
        if not allowed:
            continue
        
        for asset in assets:
            if asset['class'] not in allowed:
                alerts.append({
                    'alert': f"⚠️ {asset['class']} detected in {zone}",
                    'severity': 'HIGH',
                    'zone': zone,
                    'asset': asset['class'],
                    'confidence': asset.get('confidence', 0.0)
                })
    
    return alerts


def check_restricted_area(detections, restricted_zones):
    """
    Check if any assets are in restricted areas
    
    Args:
        detections: List of all detections
        restricted_zones: List of zone names that are restricted
        
    Returns:
        List of alerts for restricted area violations
    """
    alerts = []
    
    for zone in restricted_zones:
        if zone in detections:
            for asset in detections[zone]:
                alerts.append({
                    'alert': f"🚫 {asset['class']} in restricted area: {zone}",
                    'severity': 'CRITICAL',
                    'zone': zone,
                    'asset': asset['class'],
                    'confidence': asset.get('confidence', 0.0)
                })
    
    return alerts


def check_unauthorized_vehicles(detections, authorized_types=None):
    """
    Check for unauthorized vehicle types
    
    Args:
        detections: List of all detections
        authorized_types: List of authorized vehicle classes
        
    Returns:
        List of alerts for unauthorized vehicles
    """
    if authorized_types is None:
        authorized_types = ['truck', 'trailer', 'forklift', 'car']
    
    alerts = []
    
    vehicle_classes = ['truck', 'trailer', 'car', 'forklift']
    
    for det in detections:
        if det['class'] in vehicle_classes and det['class'] not in authorized_types:
            alerts.append({
                'alert': f"🚨 Unauthorized vehicle: {det['class']}",
                'severity': 'MEDIUM',
                'zone': 'Unknown',
                'asset': det['class'],
                'confidence': det.get('confidence', 0.0)
            })
    
    return alerts


def generate_alert_report(alerts, image_name=""):
    """
    Generate a formatted alert report
    
    Args:
        alerts: List of alert dicts
        image_name: Name of the source image
        
    Returns:
        Formatted alert report string
    """
    if not alerts:
        return f"✅ No anomalies detected in {image_name}"
    
    report = f"\n🚨 ANOMALY REPORT - {image_name}\n"
    report += "=" * 50 + "\n"
    report += f"Total Alerts: {len(alerts)}\n\n"
    
    # Group by severity
    by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for alert in alerts:
        severity = alert.get('severity', 'LOW')
        by_severity[severity].append(alert)
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if by_severity[severity]:
            report += f"\n{severity} ({len(by_severity[severity])}):\n"
            for alert in by_severity[severity]:
                report += f"  - {alert['alert']}\n"
    
    return report


def main():
    """Test anomaly detection"""
    print("SmartYard Anomaly Alert System")
    print("="*50)
    
    # Sample zone assets
    sample_zone_assets = {
        'Zone_A_Truck_Bay': [
            {'class': 'truck', 'confidence': 0.9},
            {'class': 'person', 'confidence': 0.8}  # Anomaly!
        ],
        'Zone_B_Container_Stack': [
            {'class': 'container', 'confidence': 0.95}
        ]
    }
    
    alerts = check_anomalies(sample_zone_assets)
    
    print(f"\nDetected {len(alerts)} anomalies:")
    for alert in alerts:
        print(f"  [{alert['severity']}] {alert['alert']}")
    
    print("\nTo use anomaly detection:")
    print("  from anomaly import check_anomalies")
    print("  alerts = check_anomalies(zone_assets)")


if __name__ == "__main__":
    main()
