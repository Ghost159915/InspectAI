FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Gradio listens on 7860 by default
EXPOSE 7860

# Set env so Gradio is accessible from outside the container
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "main.py"]
