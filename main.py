# AI Insights Dashboard - Backend Production Dockerfile
# ====================================================
#
# Multi-stage production build for FastAPI backend with:
# - Optimized Python runtime with security updates
# - Multi-stage build for smaller image size
# - Non-root user for security
# - Health checks and monitoring
# - Production WSGI server (Gunicorn + Uvicorn)
# - Comprehensive logging and error handling
#
# Build: docker build -f backend/Dockerfile.prod -t ai-insights-api:latest ./backend
# Run:   docker run -p 8000:8000 ai-insights-api:latest
#
# Author: AI Insights Team
# Version: 1.0.0

# =============================================================================
# STAGE 1: Build Dependencies
# =============================================================================
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# STAGE 2: Production Runtime
# =============================================================================
FROM python:3.11-slim-bullseye as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=
