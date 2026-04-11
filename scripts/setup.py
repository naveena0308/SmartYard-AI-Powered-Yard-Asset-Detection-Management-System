import os
import sys
import time
import torch
from pathlib import Path

def print_separator():
    print("\n" + "="*60)

def verify_directories():
    print("Checking Directory Structure...")
    required_dirs = [
        'data/raw', 'data/processed', 'data/annotations', 'data/test_images',
        'models', 'outputs/annotated', 'outputs/reports', 'outputs/heatmaps',
        'src/core', 'src/analytics', 'src/utils', 'scripts'
    ]
    missing = []
    for d in required_dirs:
        if os.path.exists(d):
            print(f"   [OK] {d}")
        else:
            print(f"   [MISSING] {d}")
            missing.append(d)
    return len(missing) == 0

def verify_libraries():
    print("\nChecking Python Libraries...")
    libraries = ['ultralytics', 'cv2', 'pandas', 'numpy', 'streamlit', 'roboflow']
    missing = []
    for lib in libraries:
        try:
            if lib == 'cv2':
                import cv2
            else:
                __import__(lib)
            print(f"   [OK] {lib}")
        except ImportError:
            print(f"   [MISSING] {lib}")
            missing.append(lib)
    return len(missing) == 0

def verify_gpu():
    print("\nChecking GPU Status...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   [OK] CUDA Available: {cuda_available}")
        print(f"   [OK] GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   [OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   [INFO] CUDA Not Available. Using CPU (this will be slower).")
    return cuda_available

def download_baseline_model():
    print("\nDownloading Baseline YOLOv8s Model...")
    from ultralytics import YOLO
    import shutil
    
    target_path = 'models/best.pt'
    if os.path.exists(target_path):
        print(f"   [OK] Model already exists at {target_path}")
        return YOLO(target_path)
    
    # Download yolov8s.pt
    model = YOLO('yolov8s.pt')
    os.makedirs('models', exist_ok=True)
    
    # YOLO downloads to the current working directory, let's move it
    if os.path.exists('yolov8s.pt'):
        shutil.move('yolov8s.pt', target_path)
    elif os.path.exists('models/yolov8s.pt'):
        shutil.move('models/yolov8s.pt', target_path)
        
    print(f"   [OK] Baseline model saved to {target_path}")
    return YOLO(target_path)

def run_benchmark(model, image_path):
    print(f"\nRunning Performance Benchmark on: {image_path}")
    if not os.path.exists(image_path):
        print(f"   [ERROR] Test image not found at {image_path}")
        return
    
    # Warmup
    model.predict(image_path, conf=0.25, verbose=False)
    
    # Benchmark
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        model.predict(image_path, conf=0.25, verbose=False)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # ms
    fps = 1000 / avg_latency
    
    print(f"   [RESULT] Avg Inference Latency: {avg_latency:.2f} ms")
    print(f"   [RESULT] Throughput: {fps:.2f} FPS")
    return avg_latency, fps

def main():
    print_separator()
    print("SMARTYARD PHASE 1 VERIFICATION")
    print_separator()
    
    dirs_ok = verify_directories()
    libs_ok = verify_libraries()
    gpu_ok = verify_gpu()
    
    if not libs_ok:
        print("\nVerification Failed: Some libraries are missing. Please run 'pip install -r requirements.txt'")
        return

    model = download_baseline_model()
    
    # Check for test image
    image_path = 'data/test_images/yard_test_1.png'
    if os.path.exists(image_path):
        run_benchmark(model, image_path)
    else:
        print("\nNo test image found in data/test_images/. Skipping benchmark.")
    
    print_separator()
    if dirs_ok and libs_ok:
        print("PHASE 1 COMPLETE! You are ready for Phase 2.")
        print("   You can now mention your architecture and performance metrics on your resume.")
    else:
        print("Phase 1 partial setup. Please address the errors above.")
    print_separator()

if __name__ == "__main__":
    main()
