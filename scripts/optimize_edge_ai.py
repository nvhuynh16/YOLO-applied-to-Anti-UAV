"""
Edge AI Model Optimization Script
Creates optimized model variants for edge deployment

Optimization techniques:
- FP16 Half-Precision (2x size reduction)
- INT8 Quantization (4x size reduction)
- ONNX Export (cross-platform deployment)
- TensorRT Optimization (NVIDIA edge devices)
- TFLite Export (mobile/embedded deployment)
"""

import torch
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
import json
import shutil
from typing import Dict


class EdgeAIOptimizer:
    """Optimize YOLO models for edge deployment"""

    def __init__(self, model_path: str, output_dir: str = "outputs/edge_ai/models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.model_path))
        self.model_name = self.model_path.stem

        self.optimization_results = {
            "original_model": str(self.model_path),
            "model_name": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variants": []
        }

    def create_fp16_model(self) -> Path:
        """Create FP16 half-precision model variant"""
        print("\n" + "="*70)
        print("CREATING FP16 HALF-PRECISION MODEL")
        print("="*70)

        output_path = self.output_dir / f"{self.model_name}_fp16.pt"

        try:
            # Load model and convert to half precision
            print("\nConverting model to FP16...")
            model_half = self.model.model.half()

            # Save half-precision model
            torch.save({
                'model': model_half,
                'precision': 'fp16',
                'original_model': str(self.model_path)
            }, output_path)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            original_size_mb = self.model_path.stat().st_size / (1024 * 1024)
            reduction_pct = ((original_size_mb - file_size_mb) / original_size_mb) * 100

            print(f"\nFP16 Model Created:")
            print(f"  Output: {output_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Original: {original_size_mb:.2f} MB")
            print(f"  Reduction: {reduction_pct:.1f}%")

            self.optimization_results["variants"].append({
                "type": "fp16",
                "path": str(output_path),
                "size_mb": round(file_size_mb, 2),
                "reduction_pct": round(reduction_pct, 1),
                "expected_speedup": "1.5-2x on GPU",
                "accuracy_impact": "Minimal (<0.1%)"
            })

            return output_path

        except Exception as e:
            print(f"\nERROR creating FP16 model: {e}")
            return None

    def export_to_onnx(self, img_size: int = 640, dynamic: bool = False) -> Path:
        """Export model to ONNX format"""
        print("\n" + "="*70)
        print("EXPORTING TO ONNX FORMAT")
        print("="*70)

        output_name = f"{self.model_name}_onnx"
        if dynamic:
            output_name += "_dynamic"

        try:
            print(f"\nExporting to ONNX (imgsz={img_size}, dynamic={dynamic})...")

            # Use Ultralytics export functionality
            export_path = self.model.export(
                format='onnx',
                imgsz=img_size,
                dynamic=dynamic,
                simplify=True
            )

            # Move to output directory
            export_path_obj = Path(export_path)
            final_path = self.output_dir / f"{output_name}.onnx"

            if export_path_obj.exists():
                shutil.copy(export_path_obj, final_path)

            file_size_mb = final_path.stat().st_size / (1024 * 1024)

            print(f"\nONNX Model Exported:")
            print(f"  Output: {final_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Dynamic shapes: {dynamic}")
            print(f"  Image size: {img_size}")

            self.optimization_results["variants"].append({
                "type": "onnx",
                "path": str(final_path),
                "size_mb": round(file_size_mb, 2),
                "dynamic": dynamic,
                "img_size": img_size,
                "deployment_targets": ["ONNX Runtime", "TensorRT", "OpenVINO"],
                "accuracy_impact": "None (same as PyTorch)"
            })

            return final_path

        except Exception as e:
            print(f"\nERROR exporting to ONNX: {e}")
            return None

    def export_to_tensorrt(self, img_size: int = 640, fp16: bool = True) -> Path:
        """Export model to TensorRT format (NVIDIA GPUs)"""
        print("\n" + "="*70)
        print("EXPORTING TO TENSORRT FORMAT")
        print("="*70)

        if not torch.cuda.is_available():
            print("\nWARNING: CUDA not available. TensorRT export requires NVIDIA GPU.")
            print("Skipping TensorRT export.")
            return None

        try:
            print(f"\nExporting to TensorRT (imgsz={img_size}, fp16={fp16})...")

            # Use Ultralytics export functionality
            export_path = self.model.export(
                format='engine',
                imgsz=img_size,
                half=fp16,
                device=0
            )

            export_path_obj = Path(export_path)
            final_path = self.output_dir / f"{self.model_name}_tensorrt.engine"

            if export_path_obj.exists():
                shutil.copy(export_path_obj, final_path)

            file_size_mb = final_path.stat().st_size / (1024 * 1024)

            print(f"\nTensorRT Engine Created:")
            print(f"  Output: {final_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
            print(f"  Image size: {img_size}")

            self.optimization_results["variants"].append({
                "type": "tensorrt",
                "path": str(final_path),
                "size_mb": round(file_size_mb, 2),
                "precision": "fp16" if fp16 else "fp32",
                "img_size": img_size,
                "expected_speedup": "2-5x on NVIDIA GPUs",
                "deployment_targets": ["Jetson Nano", "Jetson Xavier", "RTX GPUs"],
                "accuracy_impact": "Minimal (<0.5%)"
            })

            return final_path

        except Exception as e:
            print(f"\nERROR exporting to TensorRT: {e}")
            print("Note: TensorRT export may require TensorRT to be installed.")
            return None

    def export_to_tflite(self, img_size: int = 640, int8: bool = False) -> Path:
        """Export model to TensorFlow Lite format"""
        print("\n" + "="*70)
        print("EXPORTING TO TENSORFLOW LITE FORMAT")
        print("="*70)

        output_name = f"{self.model_name}_tflite"
        if int8:
            output_name += "_int8"

        try:
            print(f"\nExporting to TFLite (imgsz={img_size}, int8={int8})...")

            # Use Ultralytics export functionality
            export_path = self.model.export(
                format='tflite',
                imgsz=img_size,
                int8=int8
            )

            export_path_obj = Path(export_path)
            final_path = self.output_dir / f"{output_name}.tflite"

            if export_path_obj.exists():
                shutil.copy(export_path_obj, final_path)

            file_size_mb = final_path.stat().st_size / (1024 * 1024)

            print(f"\nTFLite Model Exported:")
            print(f"  Output: {final_path}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Quantization: {'INT8' if int8 else 'FP32'}")
            print(f"  Image size: {img_size}")

            self.optimization_results["variants"].append({
                "type": "tflite",
                "path": str(final_path),
                "size_mb": round(file_size_mb, 2),
                "quantization": "int8" if int8 else "fp32",
                "img_size": img_size,
                "deployment_targets": ["Android", "iOS", "Raspberry Pi", "Edge TPU"],
                "accuracy_impact": "Minimal (1-2%)" if int8 else "None"
            })

            return final_path

        except Exception as e:
            print(f"\nERROR exporting to TFLite: {e}")
            print("Note: TFLite export may require TensorFlow to be installed.")
            return None

    def export_to_coreml(self, img_size: int = 640) -> Path:
        """Export model to CoreML format (iOS/macOS)"""
        print("\n" + "="*70)
        print("EXPORTING TO COREML FORMAT")
        print("="*70)

        try:
            print(f"\nExporting to CoreML (imgsz={img_size})...")

            # Use Ultralytics export functionality
            export_path = self.model.export(
                format='coreml',
                imgsz=img_size
            )

            export_path_obj = Path(export_path)
            final_dir = self.output_dir / f"{self.model_name}_coreml.mlpackage"

            if export_path_obj.exists():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(export_path_obj, final_dir)

            # Calculate size of CoreML package
            total_size = sum(f.stat().st_size for f in final_dir.rglob('*') if f.is_file())
            file_size_mb = total_size / (1024 * 1024)

            print(f"\nCoreML Model Exported:")
            print(f"  Output: {final_dir}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Image size: {img_size}")

            self.optimization_results["variants"].append({
                "type": "coreml",
                "path": str(final_dir),
                "size_mb": round(file_size_mb, 2),
                "img_size": img_size,
                "deployment_targets": ["iOS", "macOS", "iPadOS"],
                "accuracy_impact": "None (same as PyTorch)"
            })

            return final_dir

        except Exception as e:
            print(f"\nERROR exporting to CoreML: {e}")
            print("Note: CoreML export may require coremltools to be installed.")
            return None

    def optimize_all(self,
                    create_fp16: bool = True,
                    create_onnx: bool = True,
                    create_tensorrt: bool = False,
                    create_tflite: bool = False,
                    create_coreml: bool = False) -> Dict:
        """Create all optimization variants"""
        print("\n" + "="*70)
        print("EDGE AI MODEL OPTIMIZATION SUITE")
        print(f"Original Model: {self.model_path.name}")
        print("="*70)

        # Original model info
        original_size = self.model_path.stat().st_size / (1024 * 1024)
        print(f"\nOriginal Model Size: {original_size:.2f} MB")

        # FP16 variant
        if create_fp16:
            self.create_fp16_model()

        # ONNX export
        if create_onnx:
            self.export_to_onnx(img_size=640, dynamic=False)

        # TensorRT export
        if create_tensorrt and torch.cuda.is_available():
            self.export_to_tensorrt(img_size=640, fp16=True)

        # TFLite export
        if create_tflite:
            self.export_to_tflite(img_size=640, int8=False)
            self.export_to_tflite(img_size=640, int8=True)

        # CoreML export
        if create_coreml:
            self.export_to_coreml(img_size=640)

        return self.optimization_results

    def save_optimization_report(self, filename: str = None):
        """Save optimization results to JSON"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_report_{self.model_name}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Optimization report saved to: {output_path}")
        print(f"{'='*70}")

        return output_path

    def print_summary(self):
        """Print optimization summary"""
        print(f"\n{'='*70}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"\nCreated {len(self.optimization_results['variants'])} model variants:")

        for variant in self.optimization_results['variants']:
            print(f"\n  {variant['type'].upper()}:")
            print(f"    Path: {Path(variant['path']).name}")
            print(f"    Size: {variant['size_mb']} MB")
            if 'reduction_pct' in variant:
                print(f"    Size Reduction: {variant['reduction_pct']}%")
            if 'expected_speedup' in variant:
                print(f"    Expected Speedup: {variant['expected_speedup']}")
            if 'deployment_targets' in variant:
                print(f"    Deployment: {', '.join(variant['deployment_targets'])}")

        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Edge AI Model Optimization")
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output', type=str, default='outputs/edge_ai/models',
                       help='Output directory')
    parser.add_argument('--all', action='store_true', help='Create all variants')
    parser.add_argument('--fp16', action='store_true', help='Create FP16 variant')
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX')
    parser.add_argument('--tensorrt', action='store_true', help='Export to TensorRT')
    parser.add_argument('--tflite', action='store_true', help='Export to TFLite')
    parser.add_argument('--coreml', action='store_true', help='Export to CoreML')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = EdgeAIOptimizer(
        model_path=args.model,
        output_dir=args.output
    )

    # Determine which optimizations to run
    if args.all:
        create_fp16 = True
        create_onnx = True
        create_tensorrt = True
        create_tflite = True
        create_coreml = True
    else:
        create_fp16 = args.fp16
        create_onnx = args.onnx
        create_tensorrt = args.tensorrt
        create_tflite = args.tflite
        create_coreml = args.coreml

    # If no specific optimization selected, default to FP16 and ONNX
    if not any([create_fp16, create_onnx, create_tensorrt, create_tflite, create_coreml]):
        create_fp16 = True
        create_onnx = True

    # Run optimizations
    results = optimizer.optimize_all(
        create_fp16=create_fp16,
        create_onnx=create_onnx,
        create_tensorrt=create_tensorrt,
        create_tflite=create_tflite,
        create_coreml=create_coreml
    )

    # Print summary
    optimizer.print_summary()

    # Save report
    optimizer.save_optimization_report()


if __name__ == "__main__":
    main()
