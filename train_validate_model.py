from ultralytics import YOLO
import os
import torch
import platform

# --- 1. CONFIGURATION ---

# Path to your YOLO data configuration file (updated name)
DATA_YAML_PATH = 'LDXray.yaml'

# Pretrained YOLOv8 model weights (choose 'yolov8n.pt' for Nano version)
MODEL_WEIGHTS = 'yolov8n.pt'

# Training Hyperparameters
EPOCHS = 5              # You can increase once verified working
BATCH_SIZE = 8
IMAGE_SIZE = 640
PROJECT_NAME = 'runs/detect'
RUN_NAME = 'ldxray_yolov8n_mps3'
FRACTION = 0.0001            # 1.0 = full dataset; change to smaller (e.g., 0.1) for quick test

# --- 2. Device Setup (for Apple Silicon, esp. M1/M2/M3) ---
def get_device():
    """Determines the correct device for training (MPS for Apple Silicon, CPU otherwise)."""
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        print("✅ Apple MPS (Metal) device detected and available. Using 'mps' for training.")
        return 'mps'
    else:
        print("⚠️ Apple MPS device not available or platform is not macOS. Falling back to 'cpu'.")
        return 'cpu'

DEVICE = get_device()

# --- 3. Training Function ---
def train_ldxray_model():
    """Load model, validate data paths, and train."""
    # Check for YAML file
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: Data configuration file not found at {DATA_YAML_PATH}")
        return

    print(f"📦 Loading pre-trained model: {MODEL_WEIGHTS} on device: {DEVICE}")
    model = YOLO(MODEL_WEIGHTS)

    # Start training
    print(f"🚀 Starting training using config: {DATA_YAML_PATH}...")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        name=RUN_NAME,
        project=PROJECT_NAME,
        device=DEVICE,
        workers=2,          # helps on macOS (avoid dataloader deadlocks)
        fraction=FRACTION   # set <1.0 for testing, 1.0 for full run
    )

    print("\n✅ TRAINING COMPLETE!")
    print(f"📂 Results saved to: {os.path.join(PROJECT_NAME, RUN_NAME)}")

    # --- 4. Validation (optional but recommended) ---
    try:
        best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
        if os.path.exists(best_model_path): 
            print("\n🔍 Running final validation...")
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data=DATA_YAML_PATH, device=DEVICE)

            print(f"📊 mAP50-95 (Box): {metrics.box.map:.4f}")
            print(f"📈 mAP50 (Box): {metrics.box.map50:.4f}")
        else:
            print("⚠️ No 'best.pt' weights found — skipping validation.")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

# --- 5. Run ---
if __name__ == "__main__":
    train_ldxray_model()
