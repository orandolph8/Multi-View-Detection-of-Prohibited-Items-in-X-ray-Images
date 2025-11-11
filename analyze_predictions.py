import torch
import os
import yaml
from ultralytics import YOLO

# --- Configuration ---
PROJECT_NAME = 'runs/detect'
RUN_NAME = 'ldxray_yolov8n_mps32'  # match your actual training run name
MODEL_PATH = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
DATA_YAML_PATH = 'LDXray.yaml'   # updated YAML filename
OUTPUT_DIR = 'runs/predictions_analysis'

# --- Device Setup for Apple M-series or fallback ---
def get_device():
    """Determines the correct device for inference (MPS for M-series, CPU otherwise)."""
    if torch.backends.mps.is_available():
        print("✅ MPS available — using Apple GPU acceleration.")
        return 'mps'
    else:
        print("⚠️ MPS not available — falling back to CPU.")
        return 'cpu'

DEVICE = get_device()

# --- Main Inference Function ---
def analyze_and_predict():
    """Runs inference on test images and saves YOLO predictions."""
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"[CRITICAL] Model weights not found at: {MODEL_PATH}")
        print("Please train your model first to generate 'best.pt'.")
        return

    # Load model
    print(f"📦 Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # Load dataset configuration
    if not os.path.exists(DATA_YAML_PATH):
        print(f"[CRITICAL] YAML config not found at: {DATA_YAML_PATH}")
        return

    with open(DATA_YAML_PATH, 'r') as f:
        data_cfg = yaml.safe_load(f)

    # Determine test image directory
    dataset_root = data_cfg.get('path', '').rstrip('/')
    test_rel = data_cfg.get('test', '')
    test_images_path = os.path.join(dataset_root, test_rel, '')

    if not os.path.exists(test_images_path):
        print(f"[CRITICAL] test images not found at: {test_images_path}")
        return

    print(f"🚀 Starting inference on test set: {test_images_path}")
    print(f"💻 Device: {DEVICE}")

    # Run YOLOv8 inference
    results = model.predict(
        source=test_images_path,
        save=True,
        conf=0.05,
        iou=0.45,
        device=DEVICE,
        project=OUTPUT_DIR,
        name='test_run'
    )

    # Find most recent output folder
    run_folders = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('test_run')]
    if not run_folders:
        print("[WARNING] Inference finished, but output folder not found.")
        return

    latest_run_folder = max(
        run_folders, key=lambda x: os.path.getctime(os.path.join(OUTPUT_DIR, x))
    )
    final_output_path = os.path.join(OUTPUT_DIR, latest_run_folder)
    
    
    import cv2
    
    # --- 검출된 이미지만 저장 및 표시 ---
    for i, result in enumerate(results):
        if len(result.boxes) > 0:  # 박스가 하나라도 있으면
            # 이미지 저장
            save_path = os.path.join(OUTPUT_DIR, f"{i:05d}.jpg")
            result.save(save_dir=OUTPUT_DIR)  # YOLOv8 내장 save 사용

            # 이미지 화면 표시
            img = result.orig_img.copy()
            # OpenCV는 BGR이므로 RGB로 변환
            cv2.imshow("Detected", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyAllWindows()

    print(f"saved: {OUTPUT_DIR}")

    print("\n✅ --- ANALYSIS COMPLETE ---")
    print(f"📸 Predicted images with bounding boxes saved in:\n➡️ {final_output_path}")
    print("\n🔍 Manual Review Tips:")
    print("1. Open the predicted images and inspect each bounding box.")
    print("2. Pay special attention to:")
    print("   - False Negatives (missed objects)")
    print("   - False Positives (wrong detections)")
    print("   - Low-confidence detections (<0.3)")
    print("3. Use these to improve your dataset or relabel questionable samples.")

# --- Entry Point ---
if __name__ == "__main__":
    analyze_and_predict()
