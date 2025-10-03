"""
YOLOv8 Training Script for Anti-UAV Drone Detection
Optimized for 4GB VRAM - Conservative settings
"""
from ultralytics import YOLO
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"

print("="*70)
print("YOLOv8 TRAINING: Anti-UAV Drone Detection (Light Mode)")
print("="*70)

# Initialize YOLOv8 nano model (best for 4GB VRAM)
print("\nLoading YOLOv8n (nano) model...")
model = YOLO('yolov8n.pt')

# Train with conservative settings for 4GB VRAM
print(f"\nStarting training on {DATA_YAML}...")
print(f"Output directory: {OUTPUT_DIR}")
print("\nOptimizations for 4GB VRAM:")
print("  - Batch size: 4 (reduced from 8)")
print("  - Cache: disk (instead of RAM)")
print("  - Epochs: 15 (instead of 30)")
print("  - Image size: 640")
print()

results = model.train(
    data=str(DATA_YAML),
    epochs=15,          # Reduced for faster completion
    imgsz=640,
    batch=4,            # Reduced for 4GB VRAM
    device=0,           # CUDA device
    project=str(OUTPUT_DIR),
    name='yolov8-anti-uav-light',
    patience=10,
    save=True,
    save_period=5,      # Save checkpoint every 5 epochs
    cache='disk',       # Use disk instead of RAM
    workers=0,          # 0 workers for Windows compatibility
    verbose=True
)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nBest weights: {OUTPUT_DIR / 'yolov8-anti-uav-light' / 'weights' / 'best.pt'}")
print(f"Last weights: {OUTPUT_DIR / 'yolov8-anti-uav-light' / 'weights' / 'last.pt'}")
