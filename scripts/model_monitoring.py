"""
Model Monitoring and Drift Detection for Anti-UAV Detection

This module implements production ML monitoring including:
- Prediction logging and tracking
- Data drift detection
- Model performance drift detection
- Alerting on anomalies

Usage:
    from scripts.model_monitoring import ModelMonitor

    monitor = ModelMonitor()
    monitor.log_prediction(image, predictions, ground_truth)
    monitor.check_drift()
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance and detect drift in production

    Tracks:
    - Prediction confidence distribution
    - Detection rate over time
    - Bbox size distribution
    - Inference time
    - Data drift (input distribution changes)
    - Performance drift (accuracy degradation)
    """

    def __init__(
        self,
        log_dir: str = "outputs/monitoring",
        window_size: int = 1000,
        drift_threshold: float = 0.15
    ):
        """
        Initialize monitoring system

        Args:
            log_dir: Directory to store monitoring logs
            window_size: Number of recent predictions to track
            drift_threshold: Threshold for drift detection (KL divergence)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Circular buffers for recent predictions
        self.recent_confidences = deque(maxlen=window_size)
        self.recent_bbox_sizes = deque(maxlen=window_size)
        self.recent_inference_times = deque(maxlen=window_size)
        self.recent_detection_rates = deque(maxlen=window_size)

        # Baseline statistics (computed from initial data)
        self.baseline_confidence_dist = None
        self.baseline_bbox_size_dist = None
        self.baseline_detection_rate = None

        # Metrics storage
        self.predictions_log = []
        self.drift_alerts = []

        logger.info(f"ModelMonitor initialized with window_size={window_size}")

    def log_prediction(
        self,
        image_id: str,
        predictions: List[Dict],
        inference_time: float,
        ground_truth: Optional[List[Dict]] = None
    ):
        """
        Log a single prediction for monitoring

        Args:
            image_id: Unique identifier for the image
            predictions: List of detections [{"bbox": [x,y,w,h], "confidence": float}]
            inference_time: Time taken for inference (seconds)
            ground_truth: Optional ground truth annotations
        """
        timestamp = datetime.now().isoformat()

        # Extract metrics
        num_detections = len(predictions)
        confidences = [p["confidence"] for p in predictions] if predictions else []
        bbox_sizes = [
            p["bbox"][2] * p["bbox"][3]  # width * height
            for p in predictions
        ] if predictions else []

        # Update circular buffers
        if confidences:
            self.recent_confidences.extend(confidences)
            self.recent_bbox_sizes.extend(bbox_sizes)

        self.recent_inference_times.append(inference_time)
        self.recent_detection_rates.append(1 if num_detections > 0 else 0)

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "image_id": image_id,
            "num_detections": num_detections,
            "confidences": confidences,
            "bbox_sizes": bbox_sizes,
            "inference_time": inference_time,
            "has_detection": num_detections > 0
        }

        # Add ground truth metrics if available
        if ground_truth:
            log_entry["ground_truth"] = ground_truth
            log_entry["tp"] = self._calculate_tp(predictions, ground_truth)
            log_entry["fp"] = max(0, num_detections - log_entry["tp"])
            log_entry["fn"] = max(0, len(ground_truth) - log_entry["tp"])

        self.predictions_log.append(log_entry)

        # Periodic drift check
        if len(self.predictions_log) % 100 == 0:
            self.check_drift()

        # Save logs periodically
        if len(self.predictions_log) % 500 == 0:
            self.save_logs()

    def _calculate_tp(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> int:
        """Calculate true positives using IoU matching"""
        if not predictions or not ground_truth:
            return 0

        tp_count = 0
        matched_gt = set()

        for pred in predictions:
            pred_bbox = pred["bbox"]

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                gt_bbox = gt["bbox"]
                iou = self._calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold:
                    tp_count += 1
                    matched_gt.add(gt_idx)
                    break

        return tp_count

    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def set_baseline(self):
        """
        Set baseline distribution from current data

        Should be called after processing a representative sample
        """
        if not self.recent_confidences:
            logger.warning("No data available to set baseline")
            return

        self.baseline_confidence_dist = self._get_distribution(
            list(self.recent_confidences),
            bins=20,
            range=(0, 1)
        )

        self.baseline_bbox_size_dist = self._get_distribution(
            list(self.recent_bbox_sizes),
            bins=20
        )

        self.baseline_detection_rate = np.mean(list(self.recent_detection_rates))

        logger.info("Baseline distribution set successfully")

    def check_drift(self) -> Dict[str, Any]:
        """
        Check for data and performance drift

        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_confidence_dist is None:
            logger.warning("Baseline not set, setting now...")
            self.set_baseline()
            return {"status": "baseline_set"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "confidence_drift": self._check_distribution_drift(
                list(self.recent_confidences),
                self.baseline_confidence_dist,
                "confidence"
            ),
            "bbox_size_drift": self._check_distribution_drift(
                list(self.recent_bbox_sizes),
                self.baseline_bbox_size_dist,
                "bbox_size"
            ),
            "detection_rate_drift": self._check_rate_drift()
        }

        # Check if any drift detected
        drift_detected = any([
            results["confidence_drift"]["drift_detected"],
            results["bbox_size_drift"]["drift_detected"],
            results["detection_rate_drift"]["drift_detected"]
        ])

        if drift_detected:
            logger.warning(f"DRIFT DETECTED: {results}")
            self.drift_alerts.append(results)

        return results

    def _get_distribution(
        self,
        data: List[float],
        bins: int = 20,
        range: Optional[tuple] = None
    ) -> np.ndarray:
        """Get normalized histogram distribution"""
        if not data:
            return np.zeros(bins)

        hist, _ = np.histogram(data, bins=bins, range=range, density=True)
        return hist / (hist.sum() + 1e-10)  # Normalize

    def _check_distribution_drift(
        self,
        current_data: List[float],
        baseline_dist: np.ndarray,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Check for distribution drift using KL divergence

        Args:
            current_data: Current distribution data
            baseline_dist: Baseline distribution
            metric_name: Name of the metric being checked

        Returns:
            Dictionary with drift results
        """
        if not current_data:
            return {
                "drift_detected": False,
                "reason": "No current data"
            }

        current_dist = self._get_distribution(
            current_data,
            bins=len(baseline_dist)
        )

        # Calculate KL divergence
        kl_div = self._kl_divergence(baseline_dist, current_dist)

        drift_detected = kl_div > self.drift_threshold

        return {
            "drift_detected": drift_detected,
            "kl_divergence": float(kl_div),
            "threshold": self.drift_threshold,
            "metric": metric_name
        }

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        return np.sum(p * np.log(p / q))

    def _check_rate_drift(self) -> Dict[str, Any]:
        """Check for drift in detection rate"""
        if not self.recent_detection_rates:
            return {"drift_detected": False, "reason": "No data"}

        current_rate = np.mean(list(self.recent_detection_rates))
        rate_change = abs(current_rate - self.baseline_detection_rate)

        # Alert if detection rate changes by more than 20%
        drift_detected = rate_change > 0.20

        return {
            "drift_detected": drift_detected,
            "current_rate": float(current_rate),
            "baseline_rate": float(self.baseline_detection_rate),
            "change": float(rate_change),
            "threshold": 0.20
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current monitoring metrics

        Returns:
            Dictionary with current statistics
        """
        if not self.predictions_log:
            return {"error": "No predictions logged"}

        recent_logs = self.predictions_log[-self.window_size:]

        return {
            "total_predictions": len(self.predictions_log),
            "window_size": len(recent_logs),
            "avg_inference_time": float(np.mean(self.recent_inference_times)) if self.recent_inference_times else 0,
            "avg_confidence": float(np.mean(self.recent_confidences)) if self.recent_confidences else 0,
            "detection_rate": float(np.mean(self.recent_detection_rates)) if self.recent_detection_rates else 0,
            "drift_alerts": len(self.drift_alerts)
        }

    def save_logs(self, filename: Optional[str] = None):
        """Save prediction logs to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"

        log_path = self.log_dir / filename

        with open(log_path, 'w') as f:
            json.dump({
                "predictions": self.predictions_log,
                "drift_alerts": self.drift_alerts,
                "metrics_summary": self.get_metrics_summary()
            }, f, indent=2)

        logger.info(f"Logs saved to {log_path}")

    def export_metrics_for_prometheus(self) -> str:
        """
        Export metrics in Prometheus format

        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self.get_metrics_summary()

        prometheus_metrics = f"""
# HELP model_inference_time_seconds Average inference time
# TYPE model_inference_time_seconds gauge
model_inference_time_seconds {metrics.get('avg_inference_time', 0)}

# HELP model_detection_rate Detection rate (0-1)
# TYPE model_detection_rate gauge
model_detection_rate {metrics.get('detection_rate', 0)}

# HELP model_avg_confidence Average prediction confidence
# TYPE model_avg_confidence gauge
model_avg_confidence {metrics.get('avg_confidence', 0)}

# HELP model_drift_alerts_total Total drift alerts
# TYPE model_drift_alerts_total counter
model_drift_alerts_total {metrics.get('drift_alerts', 0)}

# HELP model_predictions_total Total predictions made
# TYPE model_predictions_total counter
model_predictions_total {metrics.get('total_predictions', 0)}
"""
        return prometheus_metrics.strip()


if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()

    # Simulate predictions
    for i in range(1500):
        predictions = [
            {
                "bbox": [100 + i*0.1, 100, 50, 50],
                "confidence": 0.85 + np.random.normal(0, 0.05)
            }
        ]
        monitor.log_prediction(
            image_id=f"frame_{i:05d}",
            predictions=predictions,
            inference_time=0.05 + np.random.normal(0, 0.01)
        )

    # Check drift
    drift_results = monitor.check_drift()
    print("Drift Results:", json.dumps(drift_results, indent=2))

    # Get summary
    summary = monitor.get_metrics_summary()
    print("\nMetrics Summary:", json.dumps(summary, indent=2))

    # Export Prometheus metrics
    print("\nPrometheus Metrics:")
    print(monitor.export_metrics_for_prometheus())
