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

# Copy requirements and install core deps (skip ultralytics — too large for cloud)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    python-multipart>=0.0.9 \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.6.0 \
    python-dotenv>=1.0.0

# Copy all source files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run from backend/ so bare imports (agent, detector, etc.) resolve correctly
WORKDIR /app/backend

# Use shell form so $PORT env var is expanded by Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}