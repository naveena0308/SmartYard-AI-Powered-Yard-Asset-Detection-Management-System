"""
Streamlit Dashboard
Interactive UI for SmartYard asset detection system
"""

import streamlit as st
import cv2
import json
import os
from PIL import Image
import tempfile

# Import SmartYard modules
from src.core.detect import run_detection
from src.analytics.annotate import annotate_image
from src.analytics.zone_mapper import load_zones, map_detections_to_zones, calculate_zone_occupancy
from src.analytics.anomaly import check_anomalies
from src.analytics.compliance import check_compliance
from src.analytics.report import generate_report
from src.analytics.gate_logger import log_gate_entry


# Page configuration
st.set_page_config(
    page_title="SmartYard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚛 SmartYard — AI Yard Asset Detection System")
st.markdown("Upload a yard image to detect assets, map zones, check compliance, and generate a report.")


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def run_full_pipeline(image_path, image_name):
    """Run the complete SmartYard pipeline"""
    # Step 1: Run detection
    with st.spinner("🔍 Running object detection..."):
        detections = run_detection(image_path)
    
    # Step 2: Annotate image
    with st.spinner("🎨 Annotating image..."):
        annotated = annotate_image(image_path, detections)
    
    # Step 3: Load zones and map detections
    zones = load_zones()
    zone_assets = map_detections_to_zones(detections, zones)
    zone_occupancy = calculate_zone_occupancy(zone_assets, zones)
    
    # Step 4: Check anomalies
    alerts = check_anomalies(zone_assets)
    
    # Step 5: Check compliance
    compliance = check_compliance(detections)
    
    # Step 6: Log gate entries
    gate_entries = log_gate_entry(detections, image_name)
    
    # Step 7: Generate report
    report = generate_report(
        image_name=image_name,
        detections=detections,
        zone_occupancy=zone_occupancy,
        alerts=alerts,
        compliance=compliance,
        gate_entries=gate_entries
    )
    
    return {
        'detections': detections,
        'annotated_image': annotated,
        'zone_assets': zone_assets,
        'zone_occupancy': zone_occupancy,
        'alerts': alerts,
        'compliance': compliance,
        'gate_entries': gate_entries,
        'report': report
    }


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Model path
    model_path = st.text_input(
        "Model Path",
        value="models/best.pt"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.info("Upload an image to begin analysis")


# Main content
uploaded = st.file_uploader(
    "📤 Upload Yard Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    # Save uploaded file
    temp_input = save_uploaded_file(uploaded)
    image_name = uploaded.name
    
    try:
        # Display columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(temp_input, use_column_width=True)
        
        # Run pipeline
        results = run_full_pipeline(temp_input, image_name)
        
        with col2:
            st.subheader("🔍 Detected Assets")
            st.image(results['annotated_image'], use_column_width=True)
        
        # Detection summary
        st.subheader("📊 Detection Summary")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Total Assets", results['report']['total_assets_detected'])
        
        with col4:
            st.metric(
                "Compliance Score",
                results['compliance']['compliance_score']
            )
        
        with col5:
            st.metric("Alerts", len(results['alerts']))
        
        # Asset count breakdown
        st.subheader("📦 Asset Count by Class")
        if results['report']['asset_count']:
            col_assets = st.columns(len(results['report']['asset_count']))
            for i, (cls, count) in enumerate(results['report']['asset_count'].items()):
                col_assets[i].metric(cls.upper(), count)
        else:
            st.info("No assets detected")
        
        # Zone occupancy
        st.subheader("🗺️ Zone Occupancy")
        if results['zone_occupancy']:
            for zone_name, occupancy in results['zone_occupancy'].items():
                with st.expander(f"{zone_name} ({occupancy['asset_count']} assets)"):
                    st.write(f"**Assets:** {', '.join(occupancy['assets']) if occupancy['assets'] else 'None'}")
        else:
            st.info("No zone mappings available")
        
        # Anomaly alerts
        st.subheader("🚨 Anomaly Alerts")
        if results['alerts']:
            for alert in results['alerts']:
                if alert['severity'] == 'CRITICAL':
                    st.error(alert['alert'])
                elif alert['severity'] == 'HIGH':
                    st.warning(alert['alert'])
                else:
                    st.info(alert['alert'])
        else:
            st.success("✅ No anomalies detected")
        
        # Safety compliance
        st.subheader("🛡️ Safety Compliance")
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Status", results['compliance']['status'])
        col_c2.metric(
            "Compliant Persons",
            f"{results['compliance']['compliant_persons']}/{results['compliance']['total_persons']}"
        )
        col_c3.metric("Score", results['compliance']['compliance_score'])
        
        # Gate entries
        if results['gate_entries']:
            st.subheader("🚪 Gate Entry Log")
            st.write(f"Logged {len(results['gate_entries'])} vehicle entries")
        
        # Download report
        st.subheader("📁 Download Report")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.download_button(
                "📥 Download JSON Report",
                json.dumps(results['report'], indent=2),
                file_name=f"{os.path.splitext(image_name)[0]}_report.json",
                mime="application/json"
            )
        
        with col_d2:
            if os.path.exists('outputs/reports/gate_log.csv'):
                with open('outputs/reports/gate_log.csv', 'r') as f:
                    st.download_button(
                        "📥 Download Gate Log CSV",
                        f.read(),
                        file_name="gate_log.csv",
                        mime="text/csv"
                    )
        
        # Raw detection data
        with st.expander("📋 View Raw Detection Data"):
            st.json(results['report'])
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.exception(e)
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)

else:
    # No image uploaded - show info
    st.info("👆 Upload a yard image to get started")
    
    # Show example pipeline
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### SmartYard Pipeline
        
        1. **🔍 Object Detection**: YOLOv8 model detects trucks, containers, forklifts, persons, etc.
        2. **🎨 Annotation**: Bounding boxes and labels are drawn on the image
        3. **🗺️ Zone Mapping**: Assets are mapped to predefined yard zones
        4. **🚨 Anomaly Detection**: Zone violations and unauthorized assets are flagged
        5. **🛡️ Compliance Check**: Safety gear (helmets, vests) compliance is verified
        6. **🚪 Gate Logging**: Vehicle entries are automatically logged
        7. **📁 Report Generation**: Comprehensive JSON/CSV reports are generated
        
        ### Supported Asset Classes
        - Truck
        - Trailer
        - Container / Freight Container
        - Forklift
        - Person
        - Helmet
        - Safety Vest
        - Car
        """)


# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using **Streamlit** | **YOLOv8** | **OpenCV** | "
    "[GitHub](https://github.com/yourusername/smartyard)"
)
