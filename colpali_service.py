"""
ColPali Embedding Microservice.

This standalone FastAPI service loads the ColPali model once into GPU memory
and provides an HTTP API for generating embeddings. Both the main server and
worker can call this service, avoiding duplicate model loading.

API:
- POST /embeddings: Generate embeddings for text or images
  - Request: {"input_type": "text" | "image", "inputs": [str, ...]}
  - Response: .npz file with embeddings

Memory efficiency: Only one model instance loads, serving all clients.
"""

import base64
import io
import logging
import os
import time
from typing import List, Literal

import numpy as np
import torch
import uvicorn

# Import ColPali model and processor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ColPali Embedding Service",
    description="Dedicated microservice for ColPali embeddings",
    version="1.0.0",
)

# Global model and processor (loaded once at startup)
colpali_model = None
colpali_processor = None
attn_mode = "unknown"  # Will be set to "flash_attention_2" or "eager" after loading


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    input_type: Literal["text", "image"]
    inputs: List[str]  # Text strings or base64-encoded images


@app.on_event("startup")
async def startup_event():
    """Load ColPali model into memory at startup."""
    global colpali_model, colpali_processor, attn_mode

    logger.info("=" * 80)
    logger.info("Starting ColPali Embedding Service with Flash Attention")
    logger.info("=" * 80)

    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device detected: {device}")

    # Check GPU memory and architecture if using CUDA
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory = gpu_props.total_memory / (1024**3)
        logger.info(f"GPU: {gpu_props.name}")
        logger.info(f"GPU memory available: {gpu_memory:.2f} GB")
        logger.info(f"CUDA compute capability: {gpu_props.major}.{gpu_props.minor}")

    # Load model with Flash Attention support
    logger.info("Loading ColQwen2.5-3B model...")
    start_time = time.time()

    # Try flash_attention_2 first, fall back to eager if not available
    attn_implementation = "flash_attention_2"
    try:
        import flash_attn

        logger.info("Flash Attention detected - using flash_attention_2")
    except ImportError:
        logger.warning("Flash Attention not found - falling back to eager mode")
        attn_implementation = "eager"

    try:
        colpali_model = ColQwen2_5.from_pretrained(
            "tsystems/colqwen2.5-3b-multilingual-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
        ).eval()

        colpali_processor = ColQwen2_5_Processor.from_pretrained("tsystems/colqwen2.5-3b-multilingual-v1.0")

        attn_mode = attn_implementation  # Store for health endpoint
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"✓ Model device: {colpali_model.device}")
        logger.info(f"✓ Attention implementation: {attn_mode}")

        # Log memory usage if CUDA
        if device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")
            logger.info(f"GPU memory reserved: {reserved:.2f} GB")

            # Performance expectations
            if attn_implementation == "flash_attention_2":
                logger.info("🚀 Flash Attention enabled - expecting ~2x speedup for long sequences")

    except Exception as e:
        logger.error(f"Failed to load ColPali model: {e}")
        raise

    logger.info("=" * 80)
    logger.info("ColPali Embedding Service ready to accept requests")
    logger.info("=" * 80)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": colpali_model is not None,
        "device": str(colpali_model.device) if colpali_model else "unknown",
        "attention_mode": attn_mode,
        "flash_attention_enabled": attn_mode == "flash_attention_2",
    }


@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text or images.

    Returns .npz format compatible with ColpaliApiEmbeddingModel.
    """
    if colpali_model is None or colpali_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    input_type = request.input_type
    inputs = request.inputs
    count = len(inputs)

    logger.info(f"Received request: {count} {input_type} inputs")

    try:
        # Process inputs based on type
        if input_type == "image":
            # Decode base64 images
            images = []
            for idx, base64_str in enumerate(inputs):
                try:
                    # Remove data URL prefix if present
                    if base64_str.startswith("data:"):
                        base64_str = base64_str.split(",", 1)[1]

                    image_bytes = base64.b64decode(base64_str)
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to decode image {idx}: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid image at index {idx}")

            # Process images in batches to avoid OOM
            BATCH_SIZE = 1  # Process 1 page at a time for 12GB GPU (model takes ~11GB)
            all_embeddings = []

            for i in range(0, len(images), BATCH_SIZE):
                # Aggressive memory cleanup before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                batch_images = images[i : i + BATCH_SIZE]
                processed = colpali_processor.process_images(batch_images).to(colpali_model.device)

                # Generate embeddings for this batch
                with torch.no_grad():
                    batch_embeddings = colpali_model(**processed)
                    # Immediately move to CPU and convert to float32 to save GPU memory
                    all_embeddings.append(batch_embeddings.to(torch.float32).cpu())

                # Aggressive cleanup after batch
                del processed, batch_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # Concatenate all batch embeddings (already in float32 on CPU)
            embeddings = torch.cat(all_embeddings, dim=0)

        else:  # text
            # Process text queries
            processed = colpali_processor.process_queries(inputs).to(colpali_model.device)

            # Generate embeddings
            with torch.no_grad():
                embeddings = colpali_model(**processed)

            # Convert to float32 on CPU
            embeddings = embeddings.to(torch.float32).cpu()

        # Convert to numpy arrays
        embeddings_np = embeddings.numpy()

        # Build .npz response
        npz_buffer = io.BytesIO()
        npz_dict = {
            "count": np.array(count),
            "input_type": np.array(input_type),
        }

        # Add individual embeddings
        for i in range(count):
            npz_dict[f"emb_{i}"] = embeddings_np[i]

        np.savez(npz_buffer, **npz_dict)
        npz_buffer.seek(0)

        elapsed = time.time() - start_time
        logger.info(f"Generated {count} {input_type} embeddings in {elapsed:.2f}s " f"({elapsed/count:.3f}s per item)")

        return Response(content=npz_buffer.read(), media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ColPali Embedding Service",
        "version": "1.0.0",
        "model": "tsystems/colqwen2.5-3b-multilingual-v1.0",
        "endpoints": {
            "/health": "Health check",
            "/embeddings": "Generate embeddings (POST)",
        },
    }


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("COLPALI_PORT", "8001"))
    host = os.getenv("COLPALI_HOST", "0.0.0.0")

    logger.info(f"Starting ColPali service on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
