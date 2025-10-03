"""
Create model comparison visualizations
Shows: Original → YOLOv8n Pretrained (COCO) → YOLOv8n Refined
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test video
test_video = PROJECT_ROOT / "data" / "subset" / "20190925_101846_1_2" / "visible.mp4"

# Load models
print("Loading models...")
model_pretrained = YOLO('yolov8n.pt')  # COCO pretrained
refined_weights = PROJECT_ROOT / "runs" / "train" / "yolov8-anti-uav-light3" / "weights" / "best.pt"
model_refined = YOLO(str(refined_weights))

# Extract sample frames (25%, 50%, 75%)
cap = cv2.VideoCapture(str(test_video))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_positions = [0.25, 0.50, 0.75]

print(f"\nProcessing {len(sample_positions)} sample frames from {test_video.parent.name}...")

for idx, pos in enumerate(sample_positions, 1):
    frame_num = int(total_frames * pos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        print(f"Failed to read frame {frame_num}")
        continue

    print(f"\n[{idx}/3] Frame {frame_num} ({int(pos*100)}%):")

    # Original frame
    frame_original = frame.copy()

    # YOLOv8n Pretrained (COCO) detection
    results_pretrained = model_pretrained(frame, conf=0.25, verbose=False)
    frame_pretrained = frame.copy()

    pretrained_count = 0
    for box in results_pretrained[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cv2.rectangle(frame_pretrained, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green
        pretrained_count += 1

    # YOLOv8n Refined detection
    results_refined = model_refined(frame, conf=0.25, verbose=False)
    frame_refined = frame.copy()

    refined_count = 0
    for box in results_refined[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cv2.rectangle(frame_refined, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red
        refined_count += 1

    print(f"  Pretrained: {pretrained_count} detections")
    print(f"  Refined:    {refined_count} detections")

    # Add labels (centered at top)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]

    # Center text at top
    text1 = "Original"
    text_size1 = cv2.getTextSize(text1, font, 1, 2)[0]
    cv2.putText(frame_original, text1, ((frame_w - text_size1[0]) // 2, 30), font, 1, (255, 255, 255), 2)

    text2 = f"YOLOv8n Pretrained (COCO) - {pretrained_count} det."
    text_size2 = cv2.getTextSize(text2, font, 1, 2)[0]
    cv2.putText(frame_pretrained, text2, ((frame_w - text_size2[0]) // 2, 30), font, 1, (0, 255, 0), 2)

    text3 = f"YOLOv8n Refined (Anti-UAV) - {refined_count} det."
    text_size3 = cv2.getTextSize(text3, font, 1, 2)[0]
    cv2.putText(frame_refined, text3, ((frame_w - text_size3[0]) // 2, 30), font, 1, (0, 0, 255), 2)

    # Create 3-panel comparison (horizontal)
    h, w = frame.shape[:2]
    comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison[:, 0:w] = frame_original
    comparison[:, w:w*2] = frame_pretrained
    comparison[:, w*2:w*3] = frame_refined

    # Save comparison
    output_path = OUTPUT_DIR / f"comparison_frame{frame_num:04d}.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path}")

    # Also save individual frames
    cv2.imwrite(str(OUTPUT_DIR / f"frame{frame_num:04d}_original.jpg"), frame_original)
    cv2.imwrite(str(OUTPUT_DIR / f"frame{frame_num:04d}_pretrained.jpg"), frame_pretrained)
    cv2.imwrite(str(OUTPUT_DIR / f"frame{frame_num:04d}_refined.jpg"), frame_refined)

cap.release()

print(f"\n{'='*70}")
print("COMPARISON VISUALIZATIONS COMPLETE")
print(f"{'='*70}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"\nComparison images:")
for img in sorted(OUTPUT_DIR.glob("comparison_*.jpg")):
    print(f"  - {img.name}")
