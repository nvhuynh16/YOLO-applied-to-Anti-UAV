"""
Model Registry for Anti-UAV Drone Detection

This script provides a simple model versioning and registry system for tracking
trained models, their performance metrics, and deployment stages.

Following MLOps best practices:
- Version control for models
- Metadata tracking (metrics, hyperparameters, training details)
- Stage management (development, staging, production)
- Audit trail of model changes

Usage:
    # Register a new model
    python scripts/model_registry.py register \
        --model-path runs/train/yolov8n_light/weights/best.pt \
        --name yolov8n_refined \
        --version 1.0.0 \
        --metrics-file runs/train/yolov8n_light/results.csv

    # List all models
    python scripts/model_registry.py list

    # Promote model to production
    python scripts/model_registry.py promote --model-id yolov8n_refined_v1.0.0 --stage production

    # Get model info
    python scripts/model_registry.py info --model-id yolov8n_refined_v1.0.0
"""

import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class ModelRegistry:
    """Simple YAML-based model registry for experiment tracking"""

    def __init__(self, registry_path: str = "models/model_registry.yaml"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load existing registry or create new one"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_registry(self):
        """Save registry to YAML file"""
        with open(self.registry_path, 'w') as f:
            yaml.dump(self.registry, f, default_flow_style=False, sort_keys=False)

    def register_model(
        self,
        model_path: str,
        name: str,
        version: str,
        metrics: dict = None,
        hyperparameters: dict = None,
        description: str = None,
        stage: str = "development"
    ) -> str:
        """
        Register a new model in the registry

        Args:
            model_path: Path to model weights file
            name: Model name
            version: Version string (e.g., "1.0.0")
            metrics: Dictionary of performance metrics
            hyperparameters: Dictionary of training hyperparameters
            description: Optional model description
            stage: Deployment stage (development, staging, production)

        Returns:
            model_id: Unique identifier for the registered model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate model ID
        model_id = f"{name}_v{version}"

        # Create model directory
        model_dir = Path("models") / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model weights
        dest_path = model_dir / "best.pt"
        shutil.copy(model_path, dest_path)

        # Create model metadata
        metadata = {
            "name": name,
            "version": version,
            "model_id": model_id,
            "model_path": str(dest_path),
            "stage": stage,
            "framework": "YOLOv8",
            "created_at": datetime.now().isoformat(),
            "description": description or f"{name} version {version}",
        }

        if metrics:
            metadata["metrics"] = metrics

        if hyperparameters:
            metadata["hyperparameters"] = hyperparameters

        # Register model
        self.registry[model_id] = metadata
        self._save_registry()

        print(f"[OK] Model registered: {model_id}")
        print(f"  Path: {dest_path}")
        print(f"  Stage: {stage}")
        if metrics:
            print(f"  Metrics: {json.dumps(metrics, indent=4)}")

        return model_id

    def list_models(self, stage: str = None) -> list:
        """
        List all registered models

        Args:
            stage: Optional filter by stage

        Returns:
            List of model metadata dictionaries
        """
        models = list(self.registry.values())

        if stage:
            models = [m for m in models if m.get("stage") == stage]

        return models

    def get_model(self, model_id: str) -> dict:
        """Get model metadata by ID"""
        if model_id not in self.registry:
            raise KeyError(f"Model not found: {model_id}")

        return self.registry[model_id]

    def promote_model(self, model_id: str, stage: str):
        """
        Promote model to a different stage

        Args:
            model_id: Model identifier
            stage: Target stage (development, staging, production)
        """
        valid_stages = ["development", "staging", "production"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")

        if model_id not in self.registry:
            raise KeyError(f"Model not found: {model_id}")

        # If promoting to production, demote any existing production models
        if stage == "production":
            for mid, metadata in self.registry.items():
                if metadata.get("stage") == "production":
                    metadata["stage"] = "staging"
                    metadata["demoted_at"] = datetime.now().isoformat()

        # Promote target model
        self.registry[model_id]["stage"] = stage
        self.registry[model_id]["promoted_at"] = datetime.now().isoformat()

        self._save_registry()

        print(f"[OK] Model promoted: {model_id} -> {stage}")

    def delete_model(self, model_id: str):
        """Delete model from registry and remove files"""
        if model_id not in self.registry:
            raise KeyError(f"Model not found: {model_id}")

        # Get model directory
        model_dir = Path("models") / model_id

        # Remove from registry
        del self.registry[model_id]
        self._save_registry()

        # Remove model files
        if model_dir.exists():
            shutil.rmtree(model_dir)

        print(f"[OK] Model deleted: {model_id}")


def parse_metrics_from_results(results_csv_path: str) -> dict:
    """Parse metrics from YOLOv8 results.csv file"""
    try:
        df = pd.read_csv(results_csv_path)
        # Get metrics from last epoch
        last_row = df.iloc[-1]

        metrics = {
            "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(last_row.get("metrics/mAP50-95(B)", 0)),
            "precision": float(last_row.get("metrics/precision(B)", 0)),
            "recall": float(last_row.get("metrics/recall(B)", 0)),
        }

        return metrics
    except Exception as e:
        print(f"Warning: Could not parse metrics from {results_csv_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Model Registry for Anti-UAV Drone Detection"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("--model-path", required=True, help="Path to model weights")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--version", required=True, help="Version string")
    register_parser.add_argument("--metrics-file", help="Path to results.csv file")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument(
        "--stage",
        default="development",
        choices=["development", "staging", "production"],
        help="Deployment stage"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.add_argument("--stage", help="Filter by stage")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get model information")
    info_parser.add_argument("--model-id", required=True, help="Model ID")

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to stage")
    promote_parser.add_argument("--model-id", required=True, help="Model ID")
    promote_parser.add_argument(
        "--stage",
        required=True,
        choices=["development", "staging", "production"],
        help="Target stage"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete model")
    delete_parser.add_argument("--model-id", required=True, help="Model ID")

    args = parser.parse_args()

    # Initialize registry
    registry = ModelRegistry()

    # Execute command
    if args.command == "register":
        # Parse metrics if provided
        metrics = None
        if args.metrics_file:
            metrics = parse_metrics_from_results(args.metrics_file)

        registry.register_model(
            model_path=args.model_path,
            name=args.name,
            version=args.version,
            metrics=metrics,
            description=args.description,
            stage=args.stage
        )

    elif args.command == "list":
        models = registry.list_models(stage=args.stage)

        if not models:
            print("No models registered")
        else:
            print(f"\nRegistered Models ({len(models)}):")
            print("-" * 80)
            for model in models:
                print(f"\nModel ID: {model['model_id']}")
                print(f"  Name: {model['name']}")
                print(f"  Version: {model['version']}")
                print(f"  Stage: {model['stage']}")
                print(f"  Created: {model['created_at']}")
                if 'metrics' in model:
                    print(f"  Metrics: {json.dumps(model['metrics'], indent=4)}")

    elif args.command == "info":
        model = registry.get_model(args.model_id)
        print(f"\nModel Information:")
        print("-" * 80)
        print(yaml.dump(model, default_flow_style=False))

    elif args.command == "promote":
        registry.promote_model(args.model_id, args.stage)

    elif args.command == "delete":
        registry.delete_model(args.model_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
