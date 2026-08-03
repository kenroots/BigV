FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all deps including ultralytics + roboflow
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    python-multipart>=0.0.9 \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.6.0 \
    python-dotenv>=1.0.0 \
    ultralytics>=8.0.0 \
    roboflow>=1.1.0

# Copy source files (needed before model download so download_model.py is available)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create models + logs directories
RUN mkdir -p /app/models /app/logs

# Download African wildlife model from Roboflow (requires ROBOFLOW_API_KEY build arg).
# Falls back to yolov8n.pt (COCO) gracefully if key is not provided.
ARG ROBOFLOW_API_KEY=""
ARG ROBOFLOW_WORKSPACE="african-wildlife-mwx4d"
ARG ROBOFLOW_PROJECT="african-wildlife-8csiv"
ARG ROBOFLOW_VERSION="1"
ENV ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY} \
    ROBOFLOW_WORKSPACE=${ROBOFLOW_WORKSPACE} \
    ROBOFLOW_PROJECT=${ROBOFLOW_PROJECT} \
    ROBOFLOW_VERSION=${ROBOFLOW_VERSION}

WORKDIR /app/backend
RUN python download_model.py || echo "Roboflow download skipped — will use YOLOv8n fallback"

# Pre-download YOLOv8n weights as fallback (used when no Roboflow model present)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose port
EXPOSE 8000

# WORKDIR already set to /app/backend above — CMD resolves correctly
# Use shell form so $PORT env var is expanded by Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}