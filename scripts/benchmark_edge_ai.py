"""
Edge AI Benchmarking Script for YOLOv8 Drone Detection
Comprehensive performance metrics for edge deployment scenarios

Metrics measured:
- Model Size & Complexity
- Inference Latency (CPU/GPU)
- Throughput (FPS)
- Memory Footprint
- Computational Complexity (FLOPs, MACs)
- Accuracy (mAP50, mAP50-95)
"""

import torch
import time
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import psutil
import platform
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EdgeAIBenchmark:
    """Comprehensive benchmarking suite for edge AI deployment"""

    def __init__(self, model_path: str, data_yaml: str, output_dir: str = "outputs/edge_ai/benchmarks"):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.model_path))
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict:
        """Get system and device information"""
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["cuda_available"] = False

        return info

    def measure_model_size(self) -> Dict:
        """Measure model size and parameter count"""
        print("\n" + "="*70)
        print("MEASURING MODEL SIZE & COMPLEXITY")
        print("="*70)

        # File size
        file_size_bytes = self.model_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Parameter count
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)

        # Model layers
        total_layers = len(list(self.model.model.modules()))

        metrics = {
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_mb, 2),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameters_millions": round(total_params / 1e6, 2),
            "total_layers": total_layers
        }

        print(f"\nModel: {self.model_path.name}")
        print(f"File Size: {metrics['file_size_mb']:.2f} MB")
        print(f"Parameters: {metrics['parameters_millions']:.2f} M ({total_params:,})")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Total Layers: {total_layers}")

        return metrics

    def measure_inference_latency(self,
                                  device: str = 'cpu',
                                  img_size: int = 640,
                                  batch_size: int = 1,
                                  warmup_iterations: int = 10,
                                  benchmark_iterations: int = 100) -> Dict:
        """Measure inference latency on specified device"""
        print(f"\n{'='*70}")
        print(f"MEASURING INFERENCE LATENCY - {device.upper()} - Image Size: {img_size}")
        print(f"{'='*70}")

        # Create dummy input
        dummy_input = torch.rand(batch_size, 3, img_size, img_size)

        if device == 'cuda' and torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            self.model.to('cuda')
        else:
            self.model.to('cpu')
            device = 'cpu'

        # Warmup
        print(f"\nWarming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input, verbose=False)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking ({benchmark_iterations} iterations)...")
        latencies = []

        for i in range(benchmark_iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = self.model(dummy_input, verbose=False)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{benchmark_iterations}")

        latencies = np.array(latencies)

        metrics = {
            "device": device,
            "img_size": img_size,
            "batch_size": batch_size,
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "fps": round(1000 / np.mean(latencies), 2)
        }

        print(f"\nResults:")
        print(f"  Mean Latency: {metrics['mean_latency_ms']:.2f} ms")
        print(f"  Median Latency: {metrics['median_latency_ms']:.2f} ms")
        print(f"  Std Dev: {metrics['std_latency_ms']:.2f} ms")
        print(f"  Min/Max: {metrics['min_latency_ms']:.2f} / {metrics['max_latency_ms']:.2f} ms")
        print(f"  P95/P99: {metrics['p95_latency_ms']:.2f} / {metrics['p99_latency_ms']:.2f} ms")
        print(f"  Throughput: {metrics['fps']:.2f} FPS")

        return metrics

    def measure_memory_usage(self, device: str = 'cpu', img_size: int = 640) -> Dict:
        """Measure memory footprint during inference"""
        print(f"\n{'='*70}")
        print(f"MEASURING MEMORY USAGE - {device.upper()}")
        print(f"{'='*70}")

        # CPU Memory
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / (1024 * 1024)

        # Create dummy input
        dummy_input = torch.rand(1, 3, img_size, img_size)

        if device == 'cuda' and torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            self.model.to('cuda')
            torch.cuda.reset_peak_memory_stats()
            baseline_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            self.model.to('cpu')
            device = 'cpu'

        # Run inference
        with torch.no_grad():
            _ = self.model(dummy_input, verbose=False)

        # Measure memory after inference
        peak_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = peak_memory_mb - baseline_memory_mb

        metrics = {
            "device": device,
            "baseline_memory_mb": round(baseline_memory_mb, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "memory_increase_mb": round(memory_increase_mb, 2)
        }

        if device == 'cuda' and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_increase_mb = peak_gpu_mb - baseline_gpu_mb
            metrics["baseline_gpu_memory_mb"] = round(baseline_gpu_mb, 2)
            metrics["peak_gpu_memory_mb"] = round(peak_gpu_mb, 2)
            metrics["gpu_memory_increase_mb"] = round(gpu_increase_mb, 2)

        print(f"\nCPU Memory:")
        print(f"  Baseline: {metrics['baseline_memory_mb']:.2f} MB")
        print(f"  Peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Increase: {metrics['memory_increase_mb']:.2f} MB")

        if 'peak_gpu_memory_mb' in metrics:
            print(f"\nGPU Memory:")
            print(f"  Baseline: {metrics['baseline_gpu_memory_mb']:.2f} MB")
            print(f"  Peak: {metrics['peak_gpu_memory_mb']:.2f} MB")
            print(f"  Increase: {metrics['gpu_memory_increase_mb']:.2f} MB")

        return metrics

    def measure_accuracy(self) -> Dict:
        """Measure model accuracy on validation set"""
        print(f"\n{'='*70}")
        print(f"MEASURING ACCURACY ON VALIDATION SET")
        print(f"{'='*70}")

        # Run validation
        results = self.model.val(data=str(self.data_yaml), verbose=True)

        metrics = {
            "mAP50": round(float(results.box.map50), 5),
            "mAP50-95": round(float(results.box.map), 5),
            "precision": round(float(results.box.mp), 5),
            "recall": round(float(results.box.mr), 5)
        }

        print(f"\nAccuracy Metrics:")
        print(f"  mAP50: {metrics['mAP50']*100:.2f}%")
        print(f"  mAP50-95: {metrics['mAP50-95']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall: {metrics['recall']*100:.2f}%")

        return metrics

    def measure_flops(self, img_size: int = 640) -> Dict:
        """Estimate FLOPs for the model"""
        print(f"\n{'='*70}")
        print(f"ESTIMATING COMPUTATIONAL COMPLEXITY")
        print(f"{'='*70}")

        try:
            from thop import profile
            dummy_input = torch.rand(1, 3, img_size, img_size)
            macs, params = profile(self.model.model, inputs=(dummy_input,), verbose=False)

            # FLOPs is approximately 2 * MACs
            flops = 2 * macs

            metrics = {
                "img_size": img_size,
                "macs": int(macs),
                "flops": int(flops),
                "macs_billions": round(macs / 1e9, 2),
                "flops_billions": round(flops / 1e9, 2)
            }

            print(f"\nComputational Complexity (Image Size: {img_size}):")
            print(f"  MACs: {metrics['macs_billions']:.2f} G")
            print(f"  FLOPs: {metrics['flops_billions']:.2f} G")

        except ImportError:
            print("\nWARNING: 'thop' package not installed. Skipping FLOPs calculation.")
            print("Install with: pip install thop")
            metrics = {
                "img_size": img_size,
                "macs": None,
                "flops": None,
                "macs_billions": None,
                "flops_billions": None,
                "note": "thop package not installed"
            }

        return metrics

    def run_comprehensive_benchmark(self,
                                   test_cpu: bool = True,
                                   test_gpu: bool = True,
                                   test_accuracy: bool = True,
                                   img_sizes: List[int] = [640, 416, 320]) -> Dict:
        """Run comprehensive benchmark suite"""
        print("\n" + "="*70)
        print("EDGE AI COMPREHENSIVE BENCHMARK")
        print(f"Model: {self.model_path.name}")
        print("="*70)

        results = {
            "model_path": str(self.model_path),
            "model_name": self.model_path.stem,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device_info": self.device_info
        }

        # Model size and complexity
        results["model_size"] = self.measure_model_size()

        # Accuracy
        if test_accuracy:
            results["accuracy"] = self.measure_accuracy()

        # Latency benchmarks
        latency_results = []

        for img_size in img_sizes:
            # CPU benchmark
            if test_cpu:
                cpu_metrics = self.measure_inference_latency(
                    device='cpu',
                    img_size=img_size,
                    benchmark_iterations=50
                )
                latency_results.append(cpu_metrics)

            # GPU benchmark
            if test_gpu and torch.cuda.is_available():
                gpu_metrics = self.measure_inference_latency(
                    device='cuda',
                    img_size=img_size,
                    benchmark_iterations=100
                )
                latency_results.append(gpu_metrics)

        results["latency_benchmarks"] = latency_results

        # Memory usage
        memory_results = []
        if test_cpu:
            memory_results.append(self.measure_memory_usage(device='cpu'))
        if test_gpu and torch.cuda.is_available():
            memory_results.append(self.measure_memory_usage(device='cuda'))

        results["memory_usage"] = memory_results

        # FLOPs calculation
        flops_results = []
        for img_size in img_sizes:
            flops_results.append(self.measure_flops(img_size=img_size))
        results["flops"] = flops_results

        # Calculate composite scores
        results["composite_scores"] = self.calculate_composite_scores(results)

        return results

    def calculate_composite_scores(self, results: Dict) -> Dict:
        """Calculate composite performance scores"""
        scores = {}

        # Speed score (based on FPS on GPU @640)
        gpu_640_results = [r for r in results['latency_benchmarks']
                          if r['device'] == 'cuda' and r['img_size'] == 640]
        if gpu_640_results:
            fps = gpu_640_results[0]['fps']
            # Normalize to 0-100 (30 FPS = 100, 1 FPS = 0)
            scores['speed_score'] = min(100, round((fps / 30) * 100, 2))

        # Accuracy score (based on mAP50-95)
        if 'accuracy' in results:
            scores['accuracy_score'] = round(results['accuracy']['mAP50-95'] * 100, 2)

        # Efficiency score (FPS per MB of model size)
        if gpu_640_results and 'model_size' in results:
            fps = gpu_640_results[0]['fps']
            model_size_mb = results['model_size']['file_size_mb']
            scores['efficiency_score'] = round(fps / model_size_mb, 2)

        # Overall score (weighted average: 40% speed, 60% accuracy)
        if 'speed_score' in scores and 'accuracy_score' in scores:
            scores['overall_score'] = round(
                0.4 * scores['speed_score'] + 0.6 * scores['accuracy_score'], 2
            )

        return scores

    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{results['model_name']}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")

        return output_path

    def create_summary_report(self, results: Dict):
        """Create a summary report of benchmark results"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY REPORT")
        print(f"{'='*70}")
        print(f"\nModel: {results['model_name']}")
        print(f"Size: {results['model_size']['file_size_mb']} MB")
        print(f"Parameters: {results['model_size']['parameters_millions']} M")

        if 'accuracy' in results:
            print(f"\nAccuracy:")
            print(f"  mAP50: {results['accuracy']['mAP50']*100:.2f}%")
            print(f"  mAP50-95: {results['accuracy']['mAP50-95']*100:.2f}%")

        print(f"\nInference Performance:")
        for bench in results['latency_benchmarks']:
            print(f"  {bench['device'].upper()} @ {bench['img_size']}x{bench['img_size']}: "
                  f"{bench['mean_latency_ms']:.2f} ms ({bench['fps']:.2f} FPS)")

        if 'composite_scores' in results:
            print(f"\nComposite Scores:")
            for score_name, score_value in results['composite_scores'].items():
                print(f"  {score_name.replace('_', ' ').title()}: {score_value}")

        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Edge AI Benchmark for YOLOv8")
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--output', type=str, default='outputs/edge_ai/benchmarks',
                       help='Output directory')
    parser.add_argument('--no-cpu', action='store_true', help='Skip CPU benchmarks')
    parser.add_argument('--no-gpu', action='store_true', help='Skip GPU benchmarks')
    parser.add_argument('--no-accuracy', action='store_true', help='Skip accuracy measurement')
    parser.add_argument('--img-sizes', type=int, nargs='+', default=[640, 416, 320],
                       help='Image sizes to test')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = EdgeAIBenchmark(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_cpu=not args.no_cpu,
        test_gpu=not args.no_gpu,
        test_accuracy=not args.no_accuracy,
        img_sizes=args.img_sizes
    )

    # Create summary report
    benchmark.create_summary_report(results)

    # Save results
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
