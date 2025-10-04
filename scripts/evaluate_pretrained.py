"""
Evaluate pretrained models on Anti-UAV validation set
Compares YOLOv8n and YOLOv9-C pretrained (COCO) performance
"""
from ultralytics import YOLO
from pathlib import Path
import json

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PRETRAINED MODEL EVALUATION: Anti-UAV Validation Set")
    print("="*70)

    results_summary = {}

    # 1. YOLOv8n Pretrained (COCO)
    print("\n[1/2] Evaluating YOLOv8n Pretrained (COCO weights)...")
    print("-" * 70)

    model_yolov8n = YOLO('yolov8n.pt')  # COCO pretrained weights

    # Run validation
    val_results_v8 = model_yolov8n.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=4,
        device=0,
        workers=0,  # Windows compatibility
        verbose=False
    )

    # Extract metrics
    results_summary['yolov8n_pretrained'] = {
        'model': 'YOLOv8n Pretrained (COCO)',
        'mAP50': float(val_results_v8.box.map50),
        'mAP50-95': float(val_results_v8.box.map),
        'precision': float(val_results_v8.box.mp),
        'recall': float(val_results_v8.box.mr)
    }

    print(f"\nYOLOv8n Pretrained Results:")
    print(f"  mAP50:    {results_summary['yolov8n_pretrained']['mAP50']:.4f}")
    print(f"  mAP50-95: {results_summary['yolov8n_pretrained']['mAP50-95']:.4f}")
    print(f"  Precision: {results_summary['yolov8n_pretrained']['precision']:.4f}")
    print(f"  Recall:   {results_summary['yolov8n_pretrained']['recall']:.4f}")

    # 2. YOLOv9-C Pretrained (COCO) - if available
    print("\n[2/2] Evaluating YOLOv9-C Pretrained (COCO weights)...")
    print("-" * 70)

    yolov9_weights = PROJECT_ROOT / "models" / "yolov9" / "yolov9-c.pt"

    if yolov9_weights.exists():
        try:
            # YOLOv9 might not be supported by ultralytics directly
            # Try loading as YOLO model first
            model_yolov9 = YOLO(str(yolov9_weights))

            val_results_v9 = model_yolov9.val(
                data=str(DATA_YAML),
                imgsz=640,
                batch=4,
                device=0,
                workers=0,  # Windows compatibility
                verbose=False
            )

            results_summary['yolov9c_pretrained'] = {
                'model': 'YOLOv9-C Pretrained (COCO)',
                'mAP50': float(val_results_v9.box.map50),
                'mAP50-95': float(val_results_v9.box.map),
                'precision': float(val_results_v9.box.mp),
                'recall': float(val_results_v9.box.mr)
            }

            print(f"\nYOLOv9-C Pretrained Results:")
            print(f"  mAP50:    {results_summary['yolov9c_pretrained']['mAP50']:.4f}")
            print(f"  mAP50-95: {results_summary['yolov9c_pretrained']['mAP50-95']:.4f}")
            print(f"  Precision: {results_summary['yolov9c_pretrained']['precision']:.4f}")
            print(f"  Recall:   {results_summary['yolov9c_pretrained']['recall']:.4f}")

        except Exception as e:
            print(f"⚠ YOLOv9-C evaluation failed: {e}")
            print("  YOLOv9 may not be compatible with ultralytics validation API")
            results_summary['yolov9c_pretrained'] = {
                'model': 'YOLOv9-C Pretrained (COCO)',
                'error': str(e),
                'note': 'Not compatible with ultralytics validation API'
            }
    else:
        print(f"⚠ YOLOv9-C weights not found at: {yolov9_weights}")
        results_summary['yolov9c_pretrained'] = {
            'model': 'YOLOv9-C Pretrained (COCO)',
            'error': 'Weights file not found'
        }

    # 3. Add YOLOv8n Refined (from training)
    print("\n[Baseline] YOLOv8n Refined (Our Training):")
    print("-" * 70)

    refined_weights = PROJECT_ROOT / "runs" / "train" / "yolov8-anti-uav-light3" / "weights" / "best.pt"

    if refined_weights.exists():
        model_refined = YOLO(str(refined_weights))

        val_results_refined = model_refined.val(
            data=str(DATA_YAML),
            imgsz=640,
            batch=4,
            device=0,
            workers=0,  # Windows compatibility
            verbose=False
        )

        results_summary['yolov8n_refined'] = {
            'model': 'YOLOv8n Refined (Fine-tuned on Anti-UAV)',
            'mAP50': float(val_results_refined.box.map50),
            'mAP50-95': float(val_results_refined.box.map),
            'precision': float(val_results_refined.box.mp),
            'recall': float(val_results_refined.box.mr)
        }

        print(f"\nYOLOv8n Refined Results:")
        print(f"  mAP50:    {results_summary['yolov8n_refined']['mAP50']:.4f}")
        print(f"  mAP50-95: {results_summary['yolov8n_refined']['mAP50-95']:.4f}")
        print(f"  Precision: {results_summary['yolov8n_refined']['precision']:.4f}")
        print(f"  Recall:   {results_summary['yolov8n_refined']['recall']:.4f}")

    # Save results
    output_file = OUTPUT_DIR / "pretrained_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<45} {'mAP50':<10} {'mAP50-95':<10}")
    print("-" * 70)

    for key, data in results_summary.items():
        if 'error' in data:
            print(f"{data['model']:<45} {'ERROR':<10} {'N/A':<10}")
        else:
            print(f"{data['model']:<45} {data['mAP50']:.4f}     {data['mAP50-95']:.4f}")

    print("\n" + "="*70)
    print(f"Results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
