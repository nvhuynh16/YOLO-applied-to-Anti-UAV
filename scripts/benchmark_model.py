"""
Performance Benchmarking for Anti-UAV Detection Models

Benchmarks:
- Inference speed (FPS)
- Latency (ms per frame)
- Throughput (images/second)
- Memory usage
- GPU utilization
- Model accuracy metrics

Usage:
    python scripts/benchmark_model.py \
        --model runs/train/yolov8-anti-uav/weights/best.pt \
        --data data/processed/dataset.yaml \
        --output outputs/benchmarks/yolov8n_refined.json
"""

import argparse
import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List
import cv2
from ultralytics import YOLO
import yaml
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Comprehensive model benchmarking suite"""

    def __init__(self, model_path: str):
        """
        Initialize benchmarking

        Args:
            model_path: Path to model weights
        """
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.results = {}

        logger.info(f"Loaded model from {model_path}")

    def benchmark_inference_speed(
        self,
        test_images: List[np.ndarray],
        warmup_runs: int = 10,
        num_runs: int = 100
    ) -> Dict:
        """
        Benchmark inference speed

        Args:
            test_images: List of test images
            warmup_runs: Number of warmup iterations
            num_runs: Number of benchmark iterations

        Returns:
            Dictionary with speed metrics
        """
        logger.info("Benchmarking inference speed...")

        # Warmup
        for _ in range(warmup_runs):
            self.model(test_images[0], verbose=False)

        # Benchmark
        times = []
        for i in range(num_runs):
            img = test_images[i % len(test_images)]

            start = time.perf_counter()
            self.model(img, verbose=False)
            end = time.perf_counter()

            times.append(end - start)

        times = np.array(times)

        return {
            "avg_inference_time_ms": float(np.mean(times) * 1000),
            "std_inference_time_ms": float(np.std(times) * 1000),
            "min_inference_time_ms": float(np.min(times) * 1000),
            "max_inference_time_ms": float(np.max(times) * 1000),
            "median_inference_time_ms": float(np.median(times) * 1000),
            "p95_inference_time_ms": float(np.percentile(times, 95) * 1000),
            "p99_inference_time_ms": float(np.percentile(times, 99) * 1000),
            "fps": float(1 / np.mean(times)),
            "throughput_images_per_sec": float(1 / np.mean(times))
        }

    def benchmark_batch_processing(
        self,
        test_images: List[np.ndarray],
        batch_sizes: List[int] = [1, 2, 4, 8, 16]
    ) -> Dict:
        """
        Benchmark batch processing performance

        Args:
            test_images: List of test images
            batch_sizes: Different batch sizes to test

        Returns:
            Dictionary with batch processing metrics
        """
        logger.info("Benchmarking batch processing...")

        results = {}

        for batch_size in batch_sizes:
            # Prepare batch
            batch = test_images[:batch_size]

            # Warmup
            for _ in range(5):
                self.model(batch, verbose=False)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                self.model(batch, verbose=False)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            images_per_sec = batch_size / avg_time

            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "avg_time_ms": float(avg_time * 1000),
                "images_per_sec": float(images_per_sec),
                "time_per_image_ms": float(avg_time * 1000 / batch_size)
            }

        return results

    def benchmark_memory_usage(self, test_image: np.ndarray) -> Dict:
        """
        Benchmark memory usage

        Args:
            test_image: Test image

        Returns:
            Dictionary with memory metrics
        """
        logger.info("Benchmarking memory usage...")

        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss / (1024 ** 2)  # MB

        # Run inference
        for _ in range(10):
            self.model(test_image, verbose=False)

        # Peak memory
        peak_memory = process.memory_info().rss / (1024 ** 2)  # MB

        return {
            "baseline_memory_mb": float(baseline_memory),
            "peak_memory_mb": float(peak_memory),
            "memory_increase_mb": float(peak_memory - baseline_memory)
        }

    def benchmark_accuracy(self, dataset_yaml: str) -> Dict:
        """
        Benchmark model accuracy on validation set

        Args:
            dataset_yaml: Path to dataset configuration

        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Benchmarking accuracy on validation set...")

        try:
            # Run validation
            metrics = self.model.val(data=dataset_yaml, verbose=False)

            return {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "f1_score": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-10))
            }
        except Exception as e:
            logger.error(f"Accuracy benchmarking failed: {e}")
            return {"error": str(e)}

    def benchmark_model_size(self) -> Dict:
        """
        Get model size and parameter count

        Returns:
            Dictionary with model size metrics
        """
        logger.info("Analyzing model size...")

        model_size_mb = self.model_path.stat().st_size / (1024 ** 2)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)

        return {
            "model_size_mb": float(model_size_mb),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_architecture": "YOLOv8"
        }

    def run_full_benchmark(
        self,
        test_images_dir: str,
        dataset_yaml: str,
        num_test_images: int = 100
    ) -> Dict:
        """
        Run complete benchmark suite

        Args:
            test_images_dir: Directory with test images
            dataset_yaml: Path to dataset configuration
            num_test_images: Number of test images to use

        Returns:
            Complete benchmark results
        """
        logger.info("Running full benchmark suite...")

        # Load test images
        test_images = self._load_test_images(test_images_dir, num_test_images)

        # Run all benchmarks
        self.results = {
            "model_info": {
                "model_path": str(self.model_path),
                "timestamp": datetime.now().isoformat()
            },
            "model_size": self.benchmark_model_size(),
            "inference_speed": self.benchmark_inference_speed(test_images),
            "batch_processing": self.benchmark_batch_processing(test_images),
            "memory_usage": self.benchmark_memory_usage(test_images[0]),
            "accuracy": self.benchmark_accuracy(dataset_yaml)
        }

        # Calculate composite scores
        self.results["composite_scores"] = self._calculate_composite_scores()

        return self.results

    def _load_test_images(self, images_dir: str, num_images: int) -> List[np.ndarray]:
        """Load test images from directory"""
        images_path = Path(images_dir)
        image_files = list(images_path.glob("*.jpg"))[:num_images]

        if not image_files:
            # Create synthetic test images if no real images available
            logger.warning("No test images found, generating synthetic images")
            return [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(num_images)]

        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)

        logger.info(f"Loaded {len(images)} test images")
        return images

    def _calculate_composite_scores(self) -> Dict:
        """Calculate composite performance scores"""
        scores = {}

        # Speed score (higher is better)
        if "inference_speed" in self.results:
            fps = self.results["inference_speed"]["fps"]
            # Normalize to 0-100 scale (30 FPS = 100)
            scores["speed_score"] = min(100, (fps / 30) * 100)

        # Accuracy score (higher is better)
        if "accuracy" in self.results:
            map50_95 = self.results["accuracy"].get("mAP50-95", 0)
            scores["accuracy_score"] = map50_95 * 100

        # Efficiency score (considering speed and model size)
        if "model_size" in self.results and "inference_speed" in self.results:
            size_mb = self.results["model_size"]["model_size_mb"]
            fps = self.results["inference_speed"]["fps"]
            # FPS per MB (higher is better)
            scores["efficiency_score"] = fps / size_mb

        # Overall score (weighted average)
        if "speed_score" in scores and "accuracy_score" in scores:
            scores["overall_score"] = (
                0.4 * scores["speed_score"] +
                0.6 * scores["accuracy_score"]
            )

        return scores

    def save_results(self, output_path: str):
        """Save benchmark results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No benchmark results available")
            return

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        if "model_size" in self.results:
            print(f"\nModel Size: {self.results['model_size']['model_size_mb']:.2f} MB")
            print(f"Parameters: {self.results['model_size']['total_parameters']:,}")

        if "inference_speed" in self.results:
            print(f"\nInference Speed:")
            print(f"  Average: {self.results['inference_speed']['avg_inference_time_ms']:.2f} ms")
            print(f"  FPS: {self.results['inference_speed']['fps']:.2f}")
            print(f"  P95 Latency: {self.results['inference_speed']['p95_inference_time_ms']:.2f} ms")

        if "accuracy" in self.results:
            print(f"\nAccuracy Metrics:")
            print(f"  mAP50-95: {self.results['accuracy']['mAP50-95']:.4f}")
            print(f"  mAP50: {self.results['accuracy']['mAP50']:.4f}")
            print(f"  Precision: {self.results['accuracy']['precision']:.4f}")
            print(f"  Recall: {self.results['accuracy']['recall']:.4f}")

        if "composite_scores" in self.results:
            print(f"\nComposite Scores:")
            print(f"  Speed Score: {self.results['composite_scores'].get('speed_score', 0):.2f}/100")
            print(f"  Accuracy Score: {self.results['composite_scores'].get('accuracy_score', 0):.2f}/100")
            print(f"  Overall Score: {self.results['composite_scores'].get('overall_score', 0):.2f}/100")

        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 model performance")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--images", default="data/processed/images/val", help="Test images directory")
    parser.add_argument("--output", default="outputs/benchmarks/results.json", help="Output JSON path")
    parser.add_argument("--num-images", type=int, default=100, help="Number of test images")

    args = parser.parse_args()

    # Run benchmark
    benchmark = ModelBenchmark(args.model)
    results = benchmark.run_full_benchmark(
        test_images_dir=args.images,
        dataset_yaml=args.data,
        num_test_images=args.num_images
    )

    # Save and print results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
