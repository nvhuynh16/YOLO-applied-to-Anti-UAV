"""
Unit tests for model registry system
Tests model registration, versioning, and stage management
"""

import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from model_registry import ModelRegistry


@pytest.fixture
def temp_registry_dir():
    """Create temporary directory for registry tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model_file(temp_registry_dir):
    """Create a mock model file for testing"""
    model_path = temp_registry_dir / "test_model.pt"
    model_path.write_text("mock model weights")
    return model_path


@pytest.fixture
def registry(temp_registry_dir):
    """Create a ModelRegistry instance with temp directory"""
    registry_path = temp_registry_dir / "model_registry.yaml"
    return ModelRegistry(str(registry_path))


class TestModelRegistryInit:
    """Test ModelRegistry initialization"""

    def test_create_new_registry(self, temp_registry_dir):
        """Test creating a new registry from scratch"""
        registry_path = temp_registry_dir / "new_registry.yaml"
        registry = ModelRegistry(str(registry_path))

        assert registry.registry == {}
        # Note: File not created until _save_registry() is called
        assert registry.registry_path == registry_path

    def test_load_existing_registry(self, temp_registry_dir):
        """Test loading an existing registry"""
        registry_path = temp_registry_dir / "existing.yaml"

        # Create existing registry
        existing_data = {
            "model_v1.0.0": {
                "name": "test_model",
                "version": "1.0.0",
                "stage": "production"
            }
        }
        with open(registry_path, 'w') as f:
            yaml.dump(existing_data, f)

        # Load it
        registry = ModelRegistry(str(registry_path))
        assert len(registry.registry) == 1
        assert "model_v1.0.0" in registry.registry


class TestModelRegistration:
    """Test model registration functionality"""

    def test_register_new_model(self, registry, mock_model_file, temp_registry_dir):
        """Test registering a new model"""
        # Setup models directory in temp location
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Change working directory temporarily (or mock the Path)
        metrics = {"mAP50": 0.95, "mAP50-95": 0.82}

        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="yolov8n_test",
            version="1.0.0",
            metrics=metrics,
            description="Test model",
            stage="development"
        )

        assert model_id == "yolov8n_test_v1.0.0"
        assert model_id in registry.registry
        assert registry.registry[model_id]["name"] == "yolov8n_test"
        assert registry.registry[model_id]["metrics"]["mAP50"] == 0.95

    def test_register_model_without_metrics(self, registry, mock_model_file):
        """Test registering a model without metrics"""
        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="yolov8n_no_metrics",
            version="1.0.0",
            stage="development"
        )

        assert model_id in registry.registry
        # When metrics is None, the key may not exist or be None
        assert registry.registry[model_id].get("metrics") is None

    def test_register_model_nonexistent_file(self, registry):
        """Test registering a model with non-existent file"""
        with pytest.raises(FileNotFoundError):
            registry.register_model(
                model_path="/nonexistent/model.pt",
                name="test",
                version="1.0.0"
            )


class TestModelVersioning:
    """Test model versioning functionality"""

    def test_multiple_versions_same_model(self, registry, temp_registry_dir):
        """Test registering multiple versions of the same model"""
        # Create mock models
        model_v1 = temp_registry_dir / "model_v1.pt"
        model_v2 = temp_registry_dir / "model_v2.pt"
        model_v1.write_text("v1")
        model_v2.write_text("v2")

        # Register v1
        id_v1 = registry.register_model(
            model_path=str(model_v1),
            name="yolov8n",
            version="1.0.0"
        )

        # Register v2
        id_v2 = registry.register_model(
            model_path=str(model_v2),
            name="yolov8n",
            version="2.0.0"
        )

        assert id_v1 == "yolov8n_v1.0.0"
        assert id_v2 == "yolov8n_v2.0.0"
        assert len(registry.registry) == 2

    def test_version_format(self, registry, mock_model_file):
        """Test version string formats"""
        test_versions = ["1.0.0", "2.1.3", "0.1.0-beta", "3.0.0-rc1"]

        for version in test_versions:
            model_id = registry.register_model(
                model_path=str(mock_model_file),
                name="test_model",
                version=version
            )
            assert f"v{version}" in model_id


class TestStageManagement:
    """Test model stage management"""

    def test_default_stage_is_development(self, registry, mock_model_file):
        """Test that default stage is development"""
        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0"
        )

        assert registry.registry[model_id]["stage"] == "development"

    def test_valid_stages(self, registry, mock_model_file):
        """Test registering models with different valid stages"""
        valid_stages = ["development", "staging", "production"]

        for stage in valid_stages:
            model_id = registry.register_model(
                model_path=str(mock_model_file),
                name=f"model_{stage}",
                version="1.0.0",
                stage=stage
            )
            assert registry.registry[model_id]["stage"] == stage


class TestMetricsTracking:
    """Test metrics tracking functionality"""

    def test_store_multiple_metrics(self, registry, mock_model_file):
        """Test storing multiple performance metrics"""
        metrics = {
            "mAP50": 0.995,
            "mAP50-95": 0.832,
            "precision": 0.998,
            "recall": 0.997,
            "f1_score": 0.997
        }

        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0",
            metrics=metrics
        )

        stored_metrics = registry.registry[model_id]["metrics"]
        assert stored_metrics == metrics

    def test_metrics_precision(self, registry, mock_model_file):
        """Test that metrics maintain precision"""
        metrics = {
            "mAP50": 0.99447123,
            "mAP50-95": 0.82332456
        }

        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0",
            metrics=metrics
        )

        stored = registry.registry[model_id]["metrics"]
        assert abs(stored["mAP50"] - 0.99447123) < 1e-8
        assert abs(stored["mAP50-95"] - 0.82332456) < 1e-8


class TestRegistryPersistence:
    """Test registry persistence to YAML"""

    def test_save_and_load_registry(self, temp_registry_dir, mock_model_file):
        """Test that registry persists across instances"""
        registry_path = temp_registry_dir / "persistent.yaml"

        # Create first registry and add model
        registry1 = ModelRegistry(str(registry_path))
        model_id = registry1.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0",
            metrics={"mAP50": 0.95}
        )

        # Create new registry instance (should load from disk)
        registry2 = ModelRegistry(str(registry_path))

        assert model_id in registry2.registry
        assert registry2.registry[model_id]["metrics"]["mAP50"] == 0.95

    def test_yaml_format_valid(self, temp_registry_dir, mock_model_file):
        """Test that saved YAML is valid and readable"""
        registry_path = temp_registry_dir / "test.yaml"
        registry = ModelRegistry(str(registry_path))

        registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0"
        )

        # Read YAML directly
        with open(registry_path, 'r') as f:
            data = yaml.safe_load(f)

        assert data is not None
        assert isinstance(data, dict)


class TestModelMetadata:
    """Test model metadata storage"""

    def test_created_at_timestamp(self, registry, mock_model_file):
        """Test that created_at timestamp is added"""
        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0"
        )

        assert "created_at" in registry.registry[model_id]
        # Should be valid ISO format timestamp
        timestamp = registry.registry[model_id]["created_at"]
        assert isinstance(timestamp, str)

    def test_description_storage(self, registry, mock_model_file):
        """Test that description is stored"""
        description = "YOLOv8n refined on Anti-UAV dataset"

        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0",
            description=description
        )

        assert registry.registry[model_id]["description"] == description

    def test_hyperparameters_storage(self, registry, mock_model_file):
        """Test that hyperparameters are stored"""
        hyperparameters = {
            "epochs": 15,
            "batch_size": 4,
            "learning_rate": 0.01,
            "optimizer": "SGD"
        }

        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="test",
            version="1.0.0",
            hyperparameters=hyperparameters
        )

        assert registry.registry[model_id]["hyperparameters"] == hyperparameters


class TestModelID:
    """Test model ID generation"""

    def test_model_id_format(self, registry, mock_model_file):
        """Test model ID follows expected format"""
        model_id = registry.register_model(
            model_path=str(mock_model_file),
            name="yolov8n_refined",
            version="1.0.0"
        )

        assert model_id == "yolov8n_refined_v1.0.0"
        assert "_v" in model_id

    def test_model_id_uniqueness(self, registry, temp_registry_dir):
        """Test that model IDs are unique"""
        model1 = temp_registry_dir / "model1.pt"
        model2 = temp_registry_dir / "model2.pt"
        model1.write_text("m1")
        model2.write_text("m2")

        id1 = registry.register_model(
            model_path=str(model1),
            name="yolov8n",
            version="1.0.0"
        )

        id2 = registry.register_model(
            model_path=str(model2),
            name="yolov8n",
            version="1.0.1"
        )

        assert id1 != id2
        assert len(registry.registry) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
