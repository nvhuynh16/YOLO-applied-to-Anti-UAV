"""
Data Preparation Script for Anti-UAV Dataset
Converts Anti-UAV annotations to YOLO format and extracts video frames
"""

import os
import json
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert Anti-UAV bbox format to YOLO format

    Args:
        bbox: [x, y, w, h] where (x,y) is top-left corner
        img_width: image width
        img_height: image height

    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    width_norm = w / img_width
    height_norm = h / img_height

    return [x_center, y_center, width_norm, height_norm]


def process_video(video_path, output_images_dir, output_labels_dir, video_name):
    """
    Extract frames from video and convert annotations to YOLO format

    Args:
        video_path: Path to video directory containing visible.mp4 and visible.json
        output_images_dir: Directory to save extracted frames
        output_labels_dir: Directory to save YOLO format labels
        video_name: Name of the video (used for frame naming)

    Returns:
        Number of frames processed
    """
    video_file = os.path.join(video_path, 'visible.mp4')
    json_file = os.path.join(video_path, 'visible.json')

    # Load annotations
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    exist = annotations['exist']
    gt_rect = annotations['gt_rect']

    # Open video
    cap = cv2.VideoCapture(video_file)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if drone exists in this frame
        if frame_count < len(exist) and exist[frame_count] == 1:
            # Save frame
            frame_filename = f"{video_name}_frame_{frame_count:05d}.jpg"
            frame_path = os.path.join(output_images_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Convert bbox to YOLO format
            bbox = gt_rect[frame_count]
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

            # Save YOLO label (class_id=0 for drone)
            label_filename = f"{video_name}_frame_{frame_count:05d}.txt"
            label_path = os.path.join(output_labels_dir, label_filename)

            with open(label_path, 'w') as f:
                # Format: class_id x_center y_center width height
                f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    return saved_count


def create_dataset_yaml(output_dir, dataset_name='anti-uav'):
    """
    Create YOLO dataset configuration file
    """
    yaml_content = f"""# Anti-UAV Drone Detection Dataset
# YOLO format

path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['drone']  # class names
"""

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created dataset.yaml at: {yaml_path}")
    return yaml_path


def prepare_dataset(source_dir, output_dir, video_list=None, split='train'):
    """
    Prepare complete dataset from Anti-UAV format to YOLO format

    Args:
        source_dir: Directory containing video folders
        output_dir: Output directory for processed dataset
        video_list: List of video names to process (None = all)
        split: Dataset split ('train', 'val', or 'test')
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images', split)
    labels_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Get list of videos to process
    if video_list is None:
        video_list = [d for d in os.listdir(source_dir)
                     if os.path.isdir(os.path.join(source_dir, d))]

    print(f"\n{'='*60}")
    print(f"Processing {len(video_list)} videos from {split} split")
    print(f"{'='*60}\n")

    total_frames = 0

    for video_name in tqdm(video_list, desc=f"Overall Progress ({split})"):
        video_path = os.path.join(source_dir, video_name)

        if not os.path.exists(os.path.join(video_path, 'visible.mp4')):
            print(f"Warning: No visible.mp4 found for {video_name}, skipping...")
            continue

        if not os.path.exists(os.path.join(video_path, 'visible.json')):
            print(f"Warning: No visible.json found for {video_name}, skipping...")
            continue

        frames_saved = process_video(video_path, images_dir, labels_dir, video_name)
        total_frames += frames_saved

    print(f"\n{split.upper()} set processed:")
    print(f"  - Videos: {len(video_list)}")
    print(f"  - Frames with drones: {total_frames}")
    print(f"  - Images saved to: {images_dir}")
    print(f"  - Labels saved to: {labels_dir}\n")

    return total_frames


if __name__ == "__main__":
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / 'data'

    # Process subset (5 videos for initial testing)
    print("="*60)
    print("ANTI-UAV DATASET PREPARATION - SUBSET (5 videos)")
    print("="*60)

    subset_videos = [
        '20190925_101846_1_2',
        '20190925_101846_1_3',
        '20190925_131530_1_4',
        '20190925_101846_1_6',
        '20190925_101846_1_7'
    ]

    source_dir = DATA_ROOT / 'subset'
    output_dir = DATA_ROOT / 'processed'

    # Process as training data
    total = prepare_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        video_list=subset_videos,
        split='train'
    )

    # Create dataset.yaml
    yaml_path = create_dataset_yaml(str(output_dir))

    print("="*60)
    print(f"DATASET PREPARATION COMPLETE!")
    print(f"Total frames extracted: {total}")
    print(f"Dataset config: {yaml_path}")
    print("="*60)
