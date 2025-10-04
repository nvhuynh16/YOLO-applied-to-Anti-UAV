"""
Extract demo frames for README
"""
import cv2
import json
from pathlib import Path

# Paths
video_name = "20190925_101846_1_2"
original_video = Path(f"data/subset/{video_name}/visible.mp4")
detected_video = Path(f"outputs/videos/{video_name}_visible_detected.mp4")
predictions = Path(f"outputs/predictions/{video_name}_visible.json")
demo_dir = Path("demo")
demo_dir.mkdir(exist_ok=True)

# Load predictions to find good frames
with open(predictions) as f:
    data = json.load(f)

# Find frames with detections (exist=1)
detection_frames = [i for i, exist in enumerate(data['exist']) if exist == 1]

# Select diverse frames: early, middle, late
selected_frames = [
    detection_frames[len(detection_frames)//4],    # 25%
    detection_frames[len(detection_frames)//2],    # 50%
    detection_frames[3*len(detection_frames)//4],  # 75%
]

print(f"Selected frames: {selected_frames}")

# Extract frames
cap_orig = cv2.VideoCapture(str(original_video))
cap_det = cv2.VideoCapture(str(detected_video))

for i, frame_idx in enumerate(selected_frames):
    # Original frame
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_orig = cap_orig.read()
    if ret:
        cv2.imwrite(str(demo_dir / f"frame_{i+1}_original.jpg"), frame_orig)

    # Detected frame
    cap_det.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_det = cap_det.read()
    if ret:
        cv2.imwrite(str(demo_dir / f"frame_{i+1}_detected.jpg"), frame_det)

    print(f"[OK] Saved frame {frame_idx} (frame_{i+1}_*.jpg)")

cap_orig.release()
cap_det.release()

# Also save first detection as main demo
cap_det = cv2.VideoCapture(str(detected_video))
cap_det.set(cv2.CAP_PROP_POS_FRAMES, selected_frames[1])  # Use middle frame
ret, frame = cap_det.read()
if ret:
    cv2.imwrite(str(demo_dir / "demo_detection.jpg"), frame)
    print(f"[OK] Saved main demo image: demo_detection.jpg")
cap_det.release()

print(f"\n[SUCCESS] Demo frames extracted to {demo_dir}/")
