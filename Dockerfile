# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory in the container
WORKDIR /app

# Create and activate virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/static /app/secrets && \
    chown -R appuser:appuser /app

# Download the Whisper model during the build
ENV MODEL_VERSION=tiny.en
RUN python -c "import whisper; model = whisper.load_model('$MODEL_VERSION', download_root='/app/models')" && \
    rm -rf /root/.cache/whisper

# Copy the application code into the container
COPY . .

# Set ownership of all files to appuser
RUN chown -R appuser:appuser /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_NO_CUDA=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV NUMEXPR_MAX_THREADS=1
ENV MALLOC_ARENA_MAX=2

# Set memory limits
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_TOP_PAD_=1
ENV PYTHONMALLOC=malloc
ENV PYTORCH_CPU_ALLOC_CONF="max_split_size_mb:64"

# Set Firebase configuration environment variables
ENV FIREBASE_SERVICE_ACCOUNT_PATH=/app/secrets/firebase-service-account.json
ENV FIREBASE_PROJECT_ID=""
ENV FIREBASE_STORAGE_BUCKET=""

# Switch to non-root user
USER appuser

# Set the PORT environment variable
ENV PORT=8080

# Expose port 8080
EXPOSE 8080

# Command to run the application with optimized settings
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --limit-concurrency 1