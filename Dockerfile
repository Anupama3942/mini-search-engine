# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PORT=5000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system security updates and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application source code
COPY . .

# Build search indexes and train models offline
RUN python build_index.py && python train_ltr.py

# Ensure correct permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose production port
EXPOSE 5000

# Health check container probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start production server
CMD ["python", "app.py"]
