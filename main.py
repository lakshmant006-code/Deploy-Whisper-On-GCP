import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import subprocess
import tempfile
import shutil
import os
import torch
import time
import gc
import psutil
import logging
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin (optional)
db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    
    # Get the path to the service account key file
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
    
    if os.path.exists(service_account_path):
        # Initialize Firebase with service account
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, {
            'projectId': os.getenv('FIREBASE_PROJECT_ID'),
            'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
        })
        db = firestore.client()
        logger.info("Firebase initialized successfully with service account")
    else:
        logger.warning(f"Firebase service account file not found at {service_account_path}")
except Exception as e:
    logger.warning(f"Firebase initialization skipped (this is OK for local development): {e}")

# Update environment variable names to match new convention
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')
FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')

# POSSIBLE VERSIONS
# https://github.com/openai/whisper/tree/main
# 'tiny.en', 'tiny', 'base.en', 'base', 'small.en',
# 'small', 'medium.en', 'medium', 'large-v1', 'large-v2',
# 'large-v3', 'large'
MODEL_VERSION = "tiny.en"  # Changed to tiny.en for minimum memory usage

# Most models use 80 mels
NUM_MELS = 80

# Memory optimization settings
DEVICE = "cpu"  # Force CPU usage to save memory
MAX_MEMORY = 256  # Reduced maximum memory usage
BATCH_SIZE = 1  # Process one audio chunk at a time

# Port configuration
PORT = 8080

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Create necessary directories
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"https://{os.getenv('FIREBASE_PROJECT_ID')}.web.app",
        f"https://{os.getenv('FIREBASE_PROJECT_ID')}-us-west1.web.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, replace with your actual domain
)

# Global variables for monitoring
TOTAL_REQUESTS = 0
ACTIVE_REQUESTS = 0

@contextmanager
def request_monitor():
    """Monitor request count and memory usage."""
    global TOTAL_REQUESTS, ACTIVE_REQUESTS
    TOTAL_REQUESTS += 1
    ACTIVE_REQUESTS += 1
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        duration = time.time() - start_time
        ACTIVE_REQUESTS -= 1
        logger.info(
            f"Request completed - Duration: {duration:.2f}s, "
            f"Memory Change: {end_memory - start_memory:.2f}MB, "
            f"Active Requests: {ACTIVE_REQUESTS}"
        )

def cleanup_memory():
    """Force garbage collection and clear memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Monitor memory usage
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if memory_usage > MAX_MEMORY:
        logger.warning(f"High memory usage detected: {memory_usage:.2f}MB")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Load the model
MODEL = None
try:
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info(f"Loading model {MODEL_VERSION}...")
    
    # Set torch to use CPU and optimize memory
    torch.set_num_threads(2)  # Reduced thread count
    torch.set_num_interop_threads(1)  # Minimize inter-op parallelism
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    MODEL = whisper.load_model(MODEL_VERSION, download_root=str(MODELS_DIR))
    MODEL = MODEL.to(DEVICE)  # Move model to CPU
    
    # Enable memory efficient options
    if hasattr(MODEL, 'encoder'):
        MODEL.encoder.gradient_checkpointing = True
    
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    logger.info(f"Model loading memory usage: {end_memory - start_memory:.2f}MB")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    try:
        # Use system temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            upload_file.file.seek(0)
            shutil.copyfileobj(upload_file.file, temp_file)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save upload file")

@app.post("/check-gpu/")
async def check_gpu():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available")
    return {"cuda": True}

@app.post("/check-ffmpeg/")
async def check_ffmpeg():
    ffmpeg = True
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        print(e)
        ffmpeg = False
    if not ffmpeg:
        raise HTTPException(status_code=400, detail="FFMPEG is not available")
    return {"ffmpeg": True}

@app.post("/check-model-in-memory/")
async def check_model_in_memory():
    """Verifies if model was loaded during docker build."""
    return {"contents": os.listdir("/app/models/")}

@app.get("/health")
async def health_check():
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "memory_usage_mb": round(memory_usage, 2),
        "active_requests": ACTIVE_REQUESTS,
        "total_requests": TOTAL_REQUESTS
    }

async def store_transcription(text: str, language: str, duration: float):
    """Store transcription result in Firestore."""
    if not db:
        logger.info("Skipping Firestore storage in local development")
        return None
        
    try:
        doc_ref = db.collection('transcriptions').document()
        doc_ref.set({
            'text': text,
            'language': language,
            'duration': duration,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Stored transcription with ID: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        logger.error(f"Failed to store transcription: {e}")
        return None

@app.post("/translate/")
async def translate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    with request_monitor():
        response = {}
        temp_filepath = None
        start_time = time.time()

        try:
            temp_filepath = save_upload_file_to_temp(file)
            
            s = time.time()
            audio = whisper.load_audio(temp_filepath)
            e = time.time()
            response["load_audio_time"] = e - s

            s = time.time()
            audio = whisper.pad_or_trim(audio)
            e = time.time()
            response["pad_audio_time"] = e - s

            s = time.time()
            mel = whisper.log_mel_spectrogram(audio, n_mels=NUM_MELS).to(DEVICE)
            e = time.time()
            response["compute_mel_features_time"] = e - s

            s = time.time()
            result = whisper.decode(MODEL, mel)
            e = time.time()
            response["inference_time"] = e - s

            response["text"] = result.text
            response["language"] = result.language
            
            # Store result in Firestore
            duration = time.time() - start_time
            doc_id = await store_transcription(result.text, result.language, duration)
            if doc_id:
                response["document_id"] = doc_id
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_memory)
            
            return response

        except Exception as e:
            logger.error(f"Error in translation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e:
                    logger.error(f"Error removing temp file: {e}")

def preprocess_audio(audio_bytes):
    """Preprocess audio bytes to optimize for Whisper."""
    try:
        # Load audio from bytes
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        
        # Convert to mono and set sample rate to 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert to WAV format in memory
        output = BytesIO()
        audio.export(output, format='wav')
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Create a single temporary file for the session
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filepath = temp_file.name
            logger.info(f"Created temporary file: {temp_filepath}")
            
            while True:
                # Receive audio chunk
                audio_chunk = await websocket.receive_bytes()
                logger.info(f"Received {len(audio_chunk)} bytes of audio data")
                
                try:
                    # Preprocess audio chunk
                    processed_audio = preprocess_audio(audio_chunk)
                    
                    # Write processed audio to file
                    with open(temp_filepath, 'wb') as f:
                        f.write(processed_audio)
                    
                    # Process audio with Whisper
                    audio = whisper.load_audio(temp_filepath)
                    audio = whisper.pad_or_trim(audio)
                    
                    # Normalize audio
                    if np.abs(audio).max() > 0:
                        audio = audio / np.abs(audio).max()
                    
                    mel = whisper.log_mel_spectrogram(audio, n_mels=NUM_MELS).to(DEVICE)
                    
                    # Use faster decoding options for real-time
                    options = whisper.DecodingOptions(
                        fp16=False,
                        language='en',  # Optimize for English
                        without_timestamps=True,  # Skip timestamp generation
                        beam_size=1,  # Use faster beam search
                        best_of=1,  # Reduce number of candidates
                        patience=1  # Reduce beam search patience
                    )
                    
                    # Detect language and transcribe
                    result = whisper.decode(MODEL, mel, options)
                    
                    # Clean up memory after processing
                    cleanup_memory()
                    
                    # Send transcription result immediately if there's text
                    text = result.text.strip()
                    if text:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "language": result.language
                        })
                        logger.info(f"Sent transcription: {text}")
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
                # Force cleanup after each chunk
                cleanup_memory()
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                logger.info(f"Removed temporary file: {temp_filepath}")
        except Exception as e:
            logger.error(f"Error removing temp file: {e}")
        await websocket.close()

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Allow eval for JSON parsing
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss: http: https:; "  # Allow WebSocket connections
            "media-src 'self' blob: mediastream:; "  # Allow media recording
            "worker-src 'self' blob:; "
            "img-src 'self' data: blob:; "
            "frame-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "  # Disable plugins
            "base-uri 'self'; "  # Restrict base tag
            "form-action 'self'; "  # Restrict form submissions
            "frame-ancestors 'none'; "  # Prevent clickjacking
            "upgrade-insecure-requests; "  # Upgrade HTTP to HTTPS
        )
        # Add additional security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
