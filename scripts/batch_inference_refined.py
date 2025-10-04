"""
Batch inference script for refined YOLOv8 model
Processes all 5 subset videos and generates refined outputs
"""
import subprocess
from pathlib import Path

# 5 selected videos from claude.md
videos = [
    "20190925_101846_1_2",  # TC-EASY with Scale Variation
    "20190925_101846_1_3",  # TC-EASY with Fast Motion, Low Resolution
    "20190925_131530_1_4",  # TC-EASY (clean baseline)
    "20190925_101846_1_6",  # TC-MID with Scale Variation
    "20190925_101846_1_7",  # TC-MID
]

PROJECT_ROOT = Path(__file__).parent.parent

# Find latest YOLOv8 weights
runs_dir = PROJECT_ROOT / "runs" / "train"
yolov8_runs = sorted(runs_dir.glob("yolov8-anti-uav*"), key=lambda p: p.stat().st_mtime)

if not yolov8_runs:
    print("ERROR: No YOLOv8 training runs found!")
    print(f"Expected in: {runs_dir}")
    exit(1)

latest_run = yolov8_runs[-1]
best_weights = latest_run / "weights" / "best.pt"

if not best_weights.exists():
    print(f"ERROR: Best weights not found: {best_weights}")
    exit(1)

print("="*70)
print("BATCH INFERENCE: Processing 5 videos with refined YOLOv8 weights")
print("="*70)
print(f"\nUsing weights: {best_weights}")
print(f"From run: {latest_run.name}")
print()

for i, video_name in enumerate(videos, 1):
    video_path = PROJECT_ROOT / "data" / "subset" / video_name / "visible.mp4"

    if not video_path.exists():
        print(f"\n[{i}/5] SKIPPED: {video_name} (file not found)")
        continue

    print(f"\n[{i}/5] Processing: {video_name}")
    print(f"        Video: {video_path}")

    # Run inference
    cmd = [
        "python",
        str(PROJECT_ROOT / "scripts" / "inference_refined.py"),
        "--video", str(video_path),
        "--weights", str(best_weights),
        "--conf", "0.25"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"        Status: SUCCESS")
    else:
        print(f"        Status: FAILED (exit code {result.returncode})")

print("\n" + "="*70)
print("BATCH INFERENCE COMPLETE")
print("="*70)
print(f"\nCheck outputs:")
print(f"  Videos: {PROJECT_ROOT / 'outputs' / 'videos'}")
print(f"  JSON:   {PROJECT_ROOT / 'outputs' / 'predictions'}")
print(f"\nRefined outputs:")
for video_name in videos:
    print(f"  - {video_name}_visible_refined.mp4")
