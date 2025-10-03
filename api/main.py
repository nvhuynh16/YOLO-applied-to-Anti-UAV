"""
FastAPI Inference Service for Anti-UAV Drone Detection

This service provides REST API endpoints for deploying the YOLOv8 drone detection model.
It follows MLOps best practices for production deployment.

Endpoints:
- POST /predict: Run inference on uploaded image/video frame
- GET /health: Service health check
- GET /model/info: Get model metadata and performance metrics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anti-UAV Drone Detection API",
    description="Production inference service for YOLOv8-based drone detection",
    version="1.0.0"
)

# Load configuration
CONFIG_PATH = Path("configs/inference_config.yaml")
MODEL_REGISTRY_PATH = Path("models/model_registry.yaml")


class Detection(BaseModel):
    """Detection result for a single object"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, width, height]


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    model_config = {"protected_namespaces": ()}  # Allow model_ prefix

    detections: List[Detection]
    inference_time_ms: float
    image_shape: List[int]
    model_version: str


class ModelInfo(BaseModel):
    """Model metadata and performance metrics"""
    model_config = {"protected_namespaces": ()}  # Allow model_ prefix

    model_name: str
    version: str
    framework: str
    metrics: Dict[str, float]
    last_updated: str


class InferenceService:
    """MLOps-ready inference service with model management"""

    def __init__(self, config_path: Path):
        """Initialize service with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.model_info = None
        self._load_model()

    def _load_config(self, config_path: Path) -> dict:
        """Load inference configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            # Return default config
            return {
                'model': {
                    'weights_path': 'runs/train/yolov8n_light/weights/best.pt',
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.45
                },
                'service': {
                    'host': '0.0.0.0',
                    'port': 8000
                }
            }

    def _load_model(self):
        """Load YOLOv8 model from registry"""
        try:
            # Load model from weights path
            weights_path = self.config['model']['weights_path']
            self.model = YOLO(weights_path)
            logger.info(f"Loaded model from {weights_path}")

            # Load model metadata from registry
            if MODEL_REGISTRY_PATH.exists():
                with open(MODEL_REGISTRY_PATH, 'r') as f:
                    registry = yaml.safe_load(f)
                    # Get the latest (production) model
                    for model_id, model_data in registry.items():
                        if model_data.get('stage') == 'production':
                            self.model_info = model_data
                            logger.info(f"Loaded model info for {model_id}")
                            break

            if self.model_info is None:
                # Default model info if registry not available
                self.model_info = {
                    'name': 'yolov8n_refined',
                    'version': '1.0.0',
                    'framework': 'YOLOv8',
                    'metrics': {
                        'mAP50': 0.995,
                        'mAP50-95': 0.832
                    },
                    'created_at': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, image: np.ndarray) -> Dict:
        """Run inference on image"""
        import time
        start_time = time.time()

        try:
            # Run inference
            conf = self.config['model']['confidence_threshold']
            iou = self.config['model']['iou_threshold']

            results = self.model.predict(
                image,
                conf=conf,
                iou=iou,
                verbose=False
            )

            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    # Convert to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1

                    detections.append({
                        'class_name': 'drone',
                        'confidence': conf,
                        'bbox': [float(x), float(y), float(w), float(h)]
                    })

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                'detections': detections,
                'inference_time_ms': inference_time,
                'image_shape': list(image.shape),
                'model_version': self.model_info['version']
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# Initialize service
try:
    service = InferenceService(CONFIG_PATH)
    logger.info("Inference service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize service: {e}")
    service = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Anti-UAV Drone Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    if service is None or service.model is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model metadata and performance metrics"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return ModelInfo(
        model_name=service.model_info.get('name', 'yolov8n_refined'),
        version=service.model_info.get('version', '1.0.0'),
        framework=service.model_info.get('framework', 'YOLOv8'),
        metrics=service.model_info.get('metrics', {}),
        last_updated=service.model_info.get('created_at', datetime.now().isoformat())
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Run inference on uploaded image

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        PredictionResponse with detections and metadata
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run inference
        result = service.predict(image)

        logger.info(f"Processed image: {file.filename}, "
                   f"detections: {len(result['detections'])}, "
                   f"time: {result['inference_time_ms']:.2f}ms")

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # Run server
    host = service.config['service']['host'] if service else '0.0.0.0'
    port = service.config['service']['port'] if service else 8000

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
