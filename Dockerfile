FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements_live.txt ./

# Install Python dependencies (base + live)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_live.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mjolnir
USER mjolnir

# Expose health check port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


# Run the pattern detector
CMD ["python", "k8s_pattern_detector.py"]