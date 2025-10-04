# Model Version Control with DVC

This directory contains trained model weights managed by DVC (Data Version Control).

## Directory Structure

```
models/
├── README.md                    # This file
├── model_registry.yaml          # Model metadata and staging info
├── yolov8n.pt.dvc              # DVC pointer to pretrained weights
└── yolov8n_refined_v1.0.0/     # Refined model version
    ├── best.pt.dvc             # DVC pointer to model weights
    └── metadata.yaml           # Model training metadata
```

## DVC Workflow

### 1. Track a New Model

When you train a new model:

```bash
# Train your model (output: runs/train/experiment/weights/best.pt)
python scripts/train_yolov8_light.py

# Register the model in the registry
python scripts/model_registry.py register \
  --model-path runs/train/experiment/weights/best.pt \
  --name yolov8n_refined \
  --version 1.1.0 \
  --stage development

# Track with DVC
dvc add models/yolov8n_refined_v1.1.0/best.pt

# Commit the .dvc file to git
git add models/yolov8n_refined_v1.1.0/best.pt.dvc .gitignore
git commit -m "Add yolov8n_refined v1.1.0 model"

# Push model to remote storage
dvc push
```

### 2. Retrieve a Model

To download a specific model version:

```bash
# Pull model from DVC remote
dvc pull models/yolov8n_refined_v1.0.0/best.pt.dvc

# Or pull all models
dvc pull
```

### 3. Model Promotion Workflow

Development → Staging → Production:

```bash
# Promote to staging
python scripts/model_registry.py promote yolov8n_refined_v1.1.0 staging

# Test in staging environment...

# Promote to production
python scripts/model_registry.py promote yolov8n_refined_v1.1.0 production
```

## DVC Remote Storage

### Local Storage (Default)

Currently configured for local storage in `dvc-storage/`:

```bash
dvc remote list
# local  dvc-storage
```

### Cloud Storage (Production)

For production, configure cloud storage:

#### AWS S3
```bash
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote region us-east-1
```

#### Google Cloud Storage
```bash
dvc remote add -d gcs gs://my-bucket/dvc-storage
```

#### Azure Blob Storage
```bash
dvc remote add -d azure azure://my-container/dvc-storage
dvc remote modify azure account_name 'myaccount'
```

## Model Versioning Best Practices

1. **Semantic Versioning**: Use MAJOR.MINOR.PATCH (e.g., 1.2.3)
   - MAJOR: Architecture changes
   - MINOR: Training data or hyperparameter changes
   - PATCH: Bug fixes or minor tweaks

2. **Always Track Metrics**: Use `model_registry.py` to store:
   - mAP scores
   - Precision/Recall
   - Training date
   - Dataset version

3. **Tag Git Commits**: Match model versions with git tags
   ```bash
   git tag -a v1.1.0 -m "YOLOv8n refined model v1.1.0"
   git push origin v1.1.0
   ```

4. **Document Changes**: Update model metadata with:
   - What changed from previous version
   - Performance improvements
   - Known issues

## Troubleshooting

### Model not found
```bash
# Check DVC cache
dvc status

# Pull from remote
dvc pull
```

### Large model files in git
```bash
# Remove from git, add to DVC
git rm --cached models/large_model.pt
dvc add models/large_model.pt
git add models/large_model.pt.dvc .gitignore
git commit -m "Track large_model.pt with DVC"
```

## Model Registry API

See `scripts/model_registry.py` for Python API:

```python
from scripts.model_registry import ModelRegistry

registry = ModelRegistry()

# Register new model
registry.register(
    model_path="path/to/model.pt",
    name="yolov8n_refined",
    version="1.1.0",
    stage="development",
    metrics={"mAP50": 0.99, "mAP50-95": 0.83}
)

# Get production model
prod_model = registry.get_production_model()
print(f"Loading: {prod_model['model_path']}")
```

## References

- [DVC Documentation](https://dvc.org/doc)
- [DVC Get Started](https://dvc.org/doc/start)
- [DVC with Cloud Storage](https://dvc.org/doc/user-guide/data-management/remote-storage)
