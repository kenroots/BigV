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
    roboflow>=1.1.0 \
    "inference-sdk>=0.9.0" \
    "git+https://github.com/openai/CLIP.git"

# Copy source files (needed before model download so download_model.py is available)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create models + logs directories
RUN mkdir -p /app/models /app/logs

# Pre-download YOLO-World + YOLOv8n weights at build time
RUN python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-worldv2.pt')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

WORKDIR /app/backend

# Expose port
EXPOSE 8000

# WORKDIR already set to /app/backend above — CMD resolves correctly
# Use shell form so $PORT env var is expanded by Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}