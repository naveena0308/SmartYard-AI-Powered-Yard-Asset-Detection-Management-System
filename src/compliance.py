"""
Safety Compliance Checker
Checks if persons have helmets and safety vests
"""


def check_compliance(detections):
    """
    Check safety compliance for persons in the image
    
    Args:
        detections: List of detection dicts with 'class' and 'bbox'
        
    Returns:
        Dict with compliance statistics
    """
    # Find persons, helmets, and vests
    persons = [d for d in detections if d['class'] == 'person']
    helmets = [d for d in detections if d['class'] == 'helmet']
    vests = [d for d in detections if d['class'] == 'safety_vest']
    
    total_persons = len(persons)
    
    if total_persons == 0:
        return {
            'total_persons': 0,
            'compliant_persons': 0,
            'non_compliant_persons': 0,
            'compliance_score': '100.0%',
            'status': '✅ PASS',
            'details': 'No persons detected'
        }
    
    # Simple compliance check: count helmets and vests
    # Note: This is a simplified version. For production, you'd want to
    # match helmets/vests to specific persons based on proximity
    compliant = min(len(helmets), len(vests))
    non_compliant = total_persons - compliant
    
    # Calculate compliance score
    score = (compliant / total_persons * 100) if total_persons > 0 else 100
    
    # Determine pass/fail (threshold: 80%)
    status = '✅ PASS' if score >= 80 else '❌ FAIL'
    
    return {
        'total_persons': total_persons,
        'compliant_persons': compliant,
        'non_compliant_persons': non_compliant,
        'compliance_score': f"{score:.1f}%",
        'status': status,
        'details': f"{compliant}/{total_persons} persons wearing proper safety gear"
    }


def check_compliance_detailed(detections, proximity_threshold=50):
    """
    Detailed compliance check matching helmets/vests to persons by proximity
    
    Args:
        detections: List of detection dicts
        proximity_threshold: Max distance to associate safety gear with person
        
    Returns:
        Dict with detailed compliance information
    """
    import math
    
    persons = [d for d in detections if d['class'] == 'person']
    helmets = [d for d in detections if d['class'] == 'helmet']
    vests = [d for d in detections if d['class'] == 'safety_vest']
    
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Check each person
    compliant_count = 0
    person_details = []
    
    for person in persons:
        person_center = get_center(person['bbox'])
        has_helmet = False
        has_vest = False
        
        # Check for nearby helmet
        for helmet in helmets:
            helmet_center = get_center(helmet['bbox'])
            if distance(person_center, helmet_center) < proximity_threshold:
                has_helmet = True
                break
        
        # Check for nearby vest
        for vest in vests:
            vest_center = get_center(vest['bbox'])
            if distance(person_center, vest_center) < proximity_threshold:
                has_vest = True
                break
        
        is_compliant = has_helmet and has_vest
        if is_compliant:
            compliant_count += 1
        
        person_details.append({
            'person_bbox': person['bbox'],
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'compliant': is_compliant
        })
    
    total_persons = len(persons)
    score = (compliant_count / total_persons * 100) if total_persons > 0 else 100
    
    return {
        'total_persons': total_persons,
        'compliant_persons': compliant_count,
        'non_compliant_persons': total_persons - compliant_count,
        'compliance_score': f"{score:.1f}%",
        'status': '✅ PASS' if score >= 80 else '❌ FAIL',
        'person_details': person_details
    }


def generate_compliance_report(compliance_result, image_name=""):
    """
    Generate formatted compliance report
    
    Args:
        compliance_result: Dict from check_compliance()
        image_name: Name of source image
        
    Returns:
        Formatted report string
    """
    report = f"\n🛡️ SAFETY COMPLIANCE REPORT - {image_name}\n"
    report += "=" * 50 + "\n"
    report += f"Status: {compliance_result['status']}\n"
    report += f"Compliance Score: {compliance_result['compliance_score']}\n"
    report += f"Total Persons: {compliance_result['total_persons']}\n"
    report += f"Compliant: {compliance_result['compliant_persons']}\n"
    
    if 'non_compliant_persons' in compliance_result:
        report += f"Non-Compliant: {compliance_result['non_compliant_persons']}\n"
    
    if 'details' in compliance_result:
        report += f"Details: {compliance_result['details']}\n"
    
    return report


def main():
    """Test compliance checker"""
    print("SmartYard Safety Compliance Checker")
    print("="*50)
    
    # Sample detections
    sample_detections = [
        {'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 150, 250]},
        {'class': 'person', 'confidence': 0.85, 'bbox': [300, 200, 350, 350]},
        {'class': 'helmet', 'confidence': 0.8, 'bbox': [110, 100, 140, 130]},
        {'class': 'safety_vest', 'confidence': 0.75, 'bbox': [105, 150, 145, 200]},
    ]
    
    result = check_compliance(sample_detections)
    
    print(generate_compliance_report(result, "test_image.jpg"))
    
    print("\nTo use compliance checker:")
    print("  from compliance import check_compliance")
    print("  result = check_compliance(detections)")


if __name__ == "__main__":
    main()
