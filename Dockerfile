# Use Python 3.8 slim image (minimum required version)
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Python 3.8
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Update pip first to ensure compatibility with Python 3.8
RUN pip install --upgrade pip==23.3.1
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs cache indexes

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if you add a web interface later)
EXPOSE 8000

# Default command
CMD ["python", "cli.py", "--help"]