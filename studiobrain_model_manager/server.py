"""
FastAPI server for StudioBrain Model Manager.

Provides OpenAI-compatible endpoints for inference plus management APIs
for VRAM monitoring, model loading/unloading, and health checks.
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from studiobrain_model_manager.config import Settings, load_config
from studiobrain_model_manager.model_manager import ModelManager
from studiobrain_model_manager.registry import ModelRegistry
from studiobrain_model_manager.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated in lifespan)
# ---------------------------------------------------------------------------
_settings: Optional[Settings] = None
_model_manager: Optional[ModelManager] = None
_model_registry: Optional[ModelRegistry] = None
_vram_monitor: Optional[VRAMMonitor] = None
_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _model_manager, _model_registry, _vram_monitor, _start_time

    _start_time = time.time()
    config_path = getattr(app.state, "config_path", None)
    _settings = load_config(config_path)
    _vram_monitor = VRAMMonitor(vram_budget_gb=_settings.vram_budget_gb)
    _model_registry = ModelRegistry(_settings)
    _model_manager = ModelManager(_settings, _model_registry)

    logger.info(f"Starting Model Manager on {_settings.host}:{_settings.port}")
    await _model_manager.initialize()

    yield

    logger.info("Shutting down Model Manager...")
    await _model_manager.cleanup()
    await _model_registry.cleanup()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="StudioBrain Model Manager",
    description="GPU inference orchestrator with smart VRAM management",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    message: Optional[Dict[str, str]] = None
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str = "cmpl-model-manager"
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: List[CompletionChoice] = []
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class EmbeddingRequest(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    input: Any  # str or List[str]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData] = []
    model: str = ""
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


class ImageProcessRequest(BaseModel):
    file_path: str
    processors: List[str] = Field(default_factory=lambda: ["ram", "florence2"])
    options: Dict[str, Any] = Field(default_factory=dict)


class ModelLoadRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - _start_time),
        "gpu_available": _settings.gpu_available if _settings else False,
        "loaded_models": list(_model_manager.loaded_models.keys()) if _model_manager else [],
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible: /v1/completions
# ---------------------------------------------------------------------------

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest):
    """OpenAI-compatible completion endpoint. Routes to the appropriate text/vision processor."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    # Determine which model to use
    model_name = req.model if req.model != "default" else "qwen_text"

    processor = await _model_manager.get_or_load_model(model_name, model_type="text")
    if not processor:
        raise HTTPException(503, f"Model '{model_name}' could not be loaded")

    # Build the prompt
    prompt = req.prompt or ""
    if req.messages:
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.messages
        )

    try:
        result = await processor.process(prompt, options={
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        })
        text = result.get("text", result.get("descriptions", {}).get("short", ""))
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, str(e))

    return CompletionResponse(
        model=model_name,
        choices=[CompletionChoice(text=text, finish_reason="stop")],
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible: /v1/embeddings
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    processor = await _model_manager.get_or_load_model("embedding_service", model_type="vector")
    if not processor:
        raise HTTPException(503, "Embedding model could not be loaded")

    texts = req.input if isinstance(req.input, list) else [req.input]

    try:
        if hasattr(processor, "embed_batch"):
            embeddings = await processor.embed_batch(texts)
        else:
            import numpy as np
            embeddings = []
            for t in texts:
                emb = await processor.embed_text(t)
                embeddings.append(emb)
            embeddings = np.array(embeddings)

        data = [
            EmbeddingData(embedding=emb.tolist(), index=i)
            for i, emb in enumerate(embeddings)
        ]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, str(e))

    return EmbeddingResponse(data=data, model=req.model)


# ---------------------------------------------------------------------------
# OpenAI-compatible: /v1/audio/transcriptions
# ---------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper"),
    language: Optional[str] = Form(None),
):
    """Whisper-compatible audio transcription endpoint."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    processor = await _model_manager.get_or_load_model("whisper", model_type="audio")
    if not processor:
        raise HTTPException(503, "Whisper model could not be loaded")

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".wav")[1])
    try:
        tmp.write(await file.read())
        tmp.close()
        result = await processor.process(tmp.name, options={"language": language})
        return {"text": result.get("text", ""), "language": result.get("language", language)}
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Custom: /v1/process/image
# ---------------------------------------------------------------------------

@app.post("/v1/process/image")
async def process_image(req: ImageProcessRequest):
    """
    Custom image processing endpoint.
    Runs one or more processors (ram, florence2, blip2) and returns structured JSON.
    """
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    results = {}
    for proc_name in req.processors:
        processor = await _model_manager.get_or_load_model(proc_name, model_type="image")
        if processor:
            try:
                result = await processor.process(req.file_path, req.options)
                results[proc_name] = result
            except Exception as e:
                results[proc_name] = {"success": False, "error": str(e)}
        else:
            results[proc_name] = {"success": False, "error": f"Could not load {proc_name}"}

    return results


# ---------------------------------------------------------------------------
# Model listing: /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    """List available models and their load status (OpenAI-compatible shape)."""
    if not _model_registry:
        raise HTTPException(503, "Registry not initialized")

    info = _model_registry.get_model_info()
    models = []
    for model_type, model_dict in info.items():
        for model_name, model_data in model_dict.items():
            models.append({
                "id": model_name,
                "object": "model",
                "owned_by": "studiobrain",
                "type": model_type,
                "loaded": model_data.get("loaded", False),
                "capabilities": model_data.get("capabilities", []),
                "size": model_data.get("size", "unknown"),
            })

    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Management: /api/models/*
# ---------------------------------------------------------------------------

@app.get("/api/models/status")
async def models_status():
    """Detailed VRAM, loaded models, and tier breakdown."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")
    return await _model_manager.get_loaded_models_status()


@app.post("/api/models/load/{name}")
async def load_model(name: str, req: ModelLoadRequest = ModelLoadRequest()):
    """Load a model by name."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    success = await _model_manager.get_or_load_model(name)
    if success:
        return {"status": "loaded", "model": name}
    raise HTTPException(500, f"Failed to load model: {name}")


@app.post("/api/models/unload/{name}")
async def unload_model(name: str):
    """Unload a model by name."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    success = await _model_manager.unload_model(name, force=True)
    return {"status": "unloaded" if success else "not_loaded", "model": name}


@app.post("/api/models/free-vram")
async def free_vram():
    """Force-unload ALL models to free VRAM (useful before external workloads)."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")
    return await _model_manager.force_unload_all()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StudioBrain Model Manager")
    parser.add_argument("--config", "-c", default=None, help="Path to models.yaml")
    parser.add_argument("--host", default=None, help="Bind host (overrides config)")
    parser.add_argument("--port", "-p", type=int, default=None, help="Bind port (overrides config)")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    # Stash config path on app.state so the lifespan can pick it up
    settings = load_config(args.config)
    host = args.host or settings.host
    port = args.port or settings.port
    app.state.config_path = args.config

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
