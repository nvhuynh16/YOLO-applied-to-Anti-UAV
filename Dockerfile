# Multi-stage Dockerfile for Anti-UAV Drone Detection API
# This follows MLOps best practices for containerized deployments

# Stage 1: Base image with Python and dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-mlops.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-mlops.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Copy model files (this will be overridden by volume mount in production)
# For local testing, you can copy a default model
# COPY runs/train/yolov8n_light/weights/best.pt ./models/best.pt

# Create necessary directories
RUN mkdir -p models runs outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
