"""
Find frames where pretrained model fails but refined model succeeds
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
test_video = PROJECT_ROOT / "data" / "subset" / "20190925_101846_1_6" / "visible.mp4"
annotations = PROJECT_ROOT / "data" / "subset" / "20190925_101846_1_6" / "visible.json"

# Load ground truth
with open(annotations, 'r') as f:
    gt_data = json.load(f)

# Load models
print("Loading models...")
model_pretrained = YOLO('yolov8n.pt')
refined_weights = PROJECT_ROOT / "runs" / "train" / "yolov8-anti-uav-light3" / "weights" / "best.pt"
model_refined = YOLO(str(refined_weights))

# Open video
cap = cv2.VideoCapture(str(test_video))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nSearching {total_frames} frames for misdetections...")

# Calculate IoU
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

candidates = []

for frame_idx in range(0, total_frames, 10):  # Check every 10th frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        continue

    # Check if drone exists in this frame
    if gt_data['exist'][frame_idx] == 0:
        continue

    gt_box = gt_data['gt_rect'][frame_idx]

    # Run pretrained
    results_pretrained = model_pretrained(frame, conf=0.25, verbose=False)
    pretrained_boxes = []
    for box in results_pretrained[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        pretrained_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])

    # Run refined
    results_refined = model_refined(frame, conf=0.25, verbose=False)
    refined_boxes = []
    for box in results_refined[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        refined_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])

    # Calculate best IoUs
    pretrained_iou = 0
    if pretrained_boxes:
        pretrained_iou = max([calculate_iou(gt_box, box) for box in pretrained_boxes])

    refined_iou = 0
    if refined_boxes:
        refined_iou = max([calculate_iou(gt_box, box) for box in refined_boxes])

    # Good candidate: pretrained DETECTS but in WRONG location, refined is correct
    # We want pretrained to show a box (IoU > 0.1) but misaligned (IoU < 0.5)
    # And refined to be accurate (IoU > 0.7)
    if refined_iou > 0.7 and 0.1 < pretrained_iou < 0.5:
        candidates.append({
            'frame': frame_idx,
            'pretrained_iou': pretrained_iou,
            'refined_iou': refined_iou,
            'pretrained_detections': len(pretrained_boxes),
            'refined_detections': len(refined_boxes)
        })
        print(f"Frame {frame_idx}: Pretrained IoU={pretrained_iou:.2f} (WRONG), Refined IoU={refined_iou:.2f} (CORRECT)")

cap.release()

print(f"\nFound {len(candidates)} candidate frames")

if candidates:
    # Sort by biggest difference
    candidates.sort(key=lambda x: x['refined_iou'] - x['pretrained_iou'], reverse=True)

    print("\nTop 5 candidates:")
    for i, cand in enumerate(candidates[:5]):
        print(f"{i+1}. Frame {cand['frame']}: Pretrained IoU={cand['pretrained_iou']:.2f}, Refined IoU={cand['refined_iou']:.2f}")
        print(f"   Difference: {cand['refined_iou'] - cand['pretrained_iou']:.2f}")

    # Save best frame
    best_frame_idx = candidates[0]['frame']
    print(f"\nBest frame: {best_frame_idx}")

    cap = cv2.VideoCapture(str(test_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ret, frame = cap.read()
    cap.release()

    # Generate comparison
    frame_original = frame.copy()

    # Pretrained
    results_pretrained = model_pretrained(frame, conf=0.25, verbose=False)
    frame_pretrained = frame.copy()
    for box in results_pretrained[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cv2.rectangle(frame_pretrained, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green

    # Refined
    results_refined = model_refined(frame, conf=0.25, verbose=False)
    frame_refined = frame.copy()
    for box in results_refined[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cv2.rectangle(frame_refined, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red

    # Ground truth
    frame_gt = frame.copy()
    x, y, w, h = gt_data['gt_rect'][best_frame_idx]
    cv2.rectangle(frame_gt, (x, y), (x+w, y+h), (255, 255, 0), 2)  # Cyan

    # Add labels (centered at top)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]

    # Center text at top
    text1 = "Original"
    text_size1 = cv2.getTextSize(text1, font, 1, 2)[0]
    cv2.putText(frame_original, text1, ((frame_w - text_size1[0]) // 2, 30), font, 1, (255, 255, 255), 2)

    text2 = f"Pretrained (IoU={candidates[0]['pretrained_iou']:.2f})"
    text_size2 = cv2.getTextSize(text2, font, 1, 2)[0]
    cv2.putText(frame_pretrained, text2, ((frame_w - text_size2[0]) // 2, 30), font, 1, (0, 255, 0), 2)

    text3 = f"Refined (IoU={candidates[0]['refined_iou']:.2f})"
    text_size3 = cv2.getTextSize(text3, font, 1, 2)[0]
    cv2.putText(frame_refined, text3, ((frame_w - text_size3[0]) // 2, 30), font, 1, (0, 0, 255), 2)

    text4 = "Ground Truth"
    text_size4 = cv2.getTextSize(text4, font, 1, 2)[0]
    cv2.putText(frame_gt, text4, ((frame_w - text_size4[0]) // 2, 30), font, 1, (255, 255, 0), 2)

    # Create 4-panel comparison
    h, w = frame.shape[:2]
    comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    comparison[0:h, 0:w] = frame_original
    comparison[0:h, w:w*2] = frame_pretrained
    comparison[h:h*2, 0:w] = frame_refined
    comparison[h:h*2, w:w*2] = frame_gt

    output_dir = PROJECT_ROOT / "demo"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"comparison_misdetection_frame{best_frame_idx:04d}.jpg"
    cv2.imwrite(str(output_path), comparison)

    # Also save 3-panel for README
    comparison_3panel = np.zeros((h, w*3, 3), dtype=np.uint8)
    comparison_3panel[:, 0:w] = frame_original
    comparison_3panel[:, w:w*2] = frame_pretrained
    comparison_3panel[:, w*2:w*3] = frame_refined

    output_3panel = output_dir / "demo_comparison.jpg"
    cv2.imwrite(str(output_3panel), comparison_3panel)

    print(f"\nSaved comparison images:")
    print(f"  - {output_path}")
    print(f"  - {output_3panel}")
else:
    print("No suitable frames found!")
