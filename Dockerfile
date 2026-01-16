# Use Python 3.11 slim image for smaller size
FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_DEVICE=cpu \
    MODEL_CACHE_DIR=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory
RUN mkdir -p /app/.cache/transformers

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: blis==0.7.11 cymem==2.0.8 murmurhash==1.0.10 preshed==3.0.9 thinc==8.2.2 && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/', timeout=5)"

# Run gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
