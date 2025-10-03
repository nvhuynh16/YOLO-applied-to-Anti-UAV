"""
Create visualization plots for README
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load predictions
video_name = "20190925_101846_1_2"
predictions_path = Path(f"outputs/predictions/{video_name}_visible.json")
with open(predictions_path) as f:
    data = json.load(f)

demo_dir = Path("demo")
demo_dir.mkdir(exist_ok=True)

# 1. Detection Timeline
plt.figure(figsize=(15, 3))
plt.plot(data['exist'], linewidth=0.5, color='#2196F3')
plt.fill_between(range(len(data['exist'])), data['exist'], alpha=0.3, color='#2196F3')
plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('Detection Status', fontsize=12)
plt.title('Drone Detection Timeline', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.yticks([0, 1], ['No Drone', 'Drone Detected'])
plt.tight_layout()
plt.savefig(demo_dir / 'detection_timeline.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved detection_timeline.png")

# 2. Bounding Box Statistics
box_areas = []
box_widths = []
box_heights = []
frame_indices = []

for i, (exist, rect) in enumerate(zip(data['exist'], data['gt_rect'])):
    if exist == 1:
        x, y, w, h = rect
        area = w * h
        box_areas.append(area)
        box_widths.append(w)
        box_heights.append(h)
        frame_indices.append(i)

if box_areas:
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Box area over time
    axes[0, 0].plot(frame_indices, box_areas, linewidth=1, color='#E91E63')
    axes[0, 0].set_xlabel('Frame Number', fontsize=11)
    axes[0, 0].set_ylabel('Bounding Box Area (pixels)', fontsize=11)
    axes[0, 0].set_title('Detection Size Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Area distribution
    axes[0, 1].hist(box_areas, bins=30, edgecolor='black', color='#4CAF50', alpha=0.7)
    axes[0, 1].set_xlabel('Bounding Box Area (pixels)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Detection Sizes', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Width vs Height scatter
    axes[1, 0].scatter(box_widths, box_heights, alpha=0.5, s=20, color='#FF9800')
    axes[1, 0].set_xlabel('Width (pixels)', fontsize=11)
    axes[1, 0].set_ylabel('Height (pixels)', fontsize=11)
    axes[1, 0].set_title('Bounding Box Dimensions', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics (text)
    axes[1, 1].axis('off')
    stats_text = f"""
    DETECTION STATISTICS

    Total Frames: {len(data['exist'])}
    Frames with Detections: {sum(data['exist'])}
    Detection Rate: {sum(data['exist'])/len(data['exist'])*100:.1f}%

    BOUNDING BOX METRICS

    Average Area: {np.mean(box_areas):.0f} px²
    Min Area: {np.min(box_areas):.0f} px²
    Max Area: {np.max(box_areas):.0f} px²

    Average Width: {np.mean(box_widths):.0f} px
    Average Height: {np.mean(box_heights):.0f} px

    Aspect Ratio (W/H): {np.mean(np.array(box_widths)/np.array(box_heights)):.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=13, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(demo_dir / 'bbox_stats.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved bbox_stats.png")

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"Detection Rate: {sum(data['exist'])}/{len(data['exist'])} frames ({sum(data['exist'])/len(data['exist'])*100:.1f}%)")
    print(f"Average Box Area: {np.mean(box_areas):.0f} px²")
    print(f"Average Dimensions: {np.mean(box_widths):.0f} x {np.mean(box_heights):.0f} px")

plt.close('all')
print(f"\n[SUCCESS] Visualizations saved to {demo_dir}/")
