"""
Inference script for refined YOLOv8 model
Processes videos and generates {video_name}_visible_refined.mp4
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import numpy as np

def run_inference(video_path, weights_path, output_dir, conf_threshold=0.25):
    """Run inference on a video and save results"""
    video_path = Path(video_path)
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Video name (without visible.mp4)
    video_name = video_path.parent.name

    # Output paths
    output_video = output_dir / f"{video_name}_visible_refined.mp4"
    output_json = output_dir.parent / "predictions" / f"{video_name}_visible_refined.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {video_name}")
    print(f"Video: {video_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {output_video}")
    print(f"{'='*70}\n")

    # Load model
    model = YOLO(weights_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # JSON structure
    predictions = {
        "video_name": video_name,
        "total_frames": total_frames,
        "detections": []
    }

    frame_idx = 0
    detections_count = 0

    print(f"Processing {total_frames} frames at {fps} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)

        # Draw bounding boxes (red)
        frame_dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Convert to Anti-UAV format (x, y, w, h)
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)

            # Draw red bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            frame_dets.append({
                "bbox": [x, y, w, h],
                "confidence": conf
            })
            detections_count += 1

        # Save frame
        out.write(frame)

        # Add to JSON
        predictions["detections"].append({
            "frame": frame_idx,
            "objects": frame_dets
        })

        frame_idx += 1

        # Progress
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  {frame_idx}/{total_frames} ({progress:.1f}%) - {detections_count} detections so far")

    cap.release()
    out.release()

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=2)

    detection_rate = (detections_count / total_frames) * 100
    print(f"\n✓ Complete!")
    print(f"  Video: {output_video}")
    print(f"  JSON: {output_json}")
    print(f"  Detections: {detections_count}/{total_frames} frames ({detection_rate:.1f}%)")

    return output_video, output_json, detections_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"

    run_inference(
        video_path=args.video,
        weights_path=args.weights,
        output_dir=OUTPUT_DIR,
        conf_threshold=args.conf
    )
