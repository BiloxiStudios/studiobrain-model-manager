"""
FastAPI server for StudioBrain Model Manager.

Provides OpenAI-compatible endpoints for inference plus management APIs
for VRAM monitoring, model loading/unloading, and health checks.
"""

import argparse
import base64
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from studiobrain_model_manager.config import Settings, load_config
from studiobrain_model_manager.litellm_registration import register_with_litellm, deregister_from_litellm
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


def _gen_id(prefix: str = "cmpl") -> str:
    """Generate a unique OpenAI-style response ID."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


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

    # Register models with LiteLLM proxy (non-blocking, retries in background)
    await register_with_litellm(_model_registry, _settings.port)

    yield

    logger.info("Shutting down Model Manager...")
    # Deregister models from LiteLLM proxy (best-effort)
    await deregister_from_litellm(_model_registry, _settings.port)
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
    id: str = Field(default_factory=lambda: _gen_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: List[CompletionChoice] = []
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage = Field(default_factory=ChatCompletionMessage)
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: List[ChatCompletionChoice] = []
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
    """Request body for /v1/process/image (JSON mode)."""
    file_path: Optional[str] = None        # local filesystem path
    image_base64: Optional[str] = None     # base64-encoded image bytes
    processors: Optional[str] = "all"      # "ram", "florence", "all"
    options: Dict[str, Any] = Field(default_factory=dict)


class ModelLoadRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ~ 4 chars)."""
    return max(1, len(text) // 4)


async def _run_text_generation(model_name: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Load the requested model and run text generation, returning the output text."""
    processor = await _model_manager.get_or_load_model(model_name, model_type="text")
    if not processor:
        raise HTTPException(503, f"Model '{model_name}' could not be loaded")

    result = await processor.process(prompt, options={
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    # Processors may return text in different keys depending on implementation
    text = result.get("text", "")
    if not text:
        text = result.get("descriptions", {}).get("short", "")
    if not text:
        text = result.get("output", "")
    return text


def _resolve_model_name(model: str) -> str:
    """Map an OpenAI-style model name to an internal model name."""
    if model in ("default", ""):
        return "qwen_text"
    return model


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
# OpenAI-compatible: /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(req: CompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    LiteLLM routes here when configured as a custom provider.
    """
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    model_name = _resolve_model_name(req.model)

    # Build the prompt from messages
    prompt = req.prompt or ""
    if req.messages:
        parts = []
        for m in req.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            # Handle content that may be a list (vision messages with text + image_url)
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = "\n".join(text_parts)
            parts.append(f"{role}: {content}")
        prompt = "\n".join(parts)

    if not prompt:
        raise HTTPException(400, "Either 'prompt' or 'messages' must be provided")

    try:
        text = await _run_text_generation(model_name, prompt, req.max_tokens, req.temperature)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(text)

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible: /v1/completions (legacy text completion)
# ---------------------------------------------------------------------------

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest):
    """OpenAI-compatible text completion endpoint."""
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    model_name = _resolve_model_name(req.model)

    # Build the prompt
    prompt = req.prompt or ""
    if req.messages:
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.messages
        )

    if not prompt:
        raise HTTPException(400, "Either 'prompt' or 'messages' must be provided")

    try:
        text = await _run_text_generation(model_name, prompt, req.max_tokens, req.temperature)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(text)

    return CompletionResponse(
        model=model_name,
        choices=[CompletionChoice(text=text, finish_reason="stop")],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
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

    # Ensure the embedding model is actually loaded (EmbeddingService uses load() not load_model())
    if hasattr(processor, "is_loaded") and not processor.is_loaded():
        if hasattr(processor, "load"):
            loaded = await processor.load()
            if not loaded:
                raise HTTPException(503, "Failed to initialize embedding model")

    texts = req.input if isinstance(req.input, list) else [req.input]

    if not texts:
        raise HTTPException(400, "Input must be a non-empty string or list of strings")

    try:
        import numpy as np

        if hasattr(processor, "embed_batch"):
            embeddings = await processor.embed_batch(texts)
        else:
            embeddings = []
            for t in texts:
                emb = await processor.embed_text(t)
                embeddings.append(emb)
            embeddings = np.array(embeddings)

        if embeddings is None:
            raise HTTPException(500, "Embedding generation returned None")

        data = [
            EmbeddingData(embedding=emb.tolist(), index=i)
            for i, emb in enumerate(embeddings)
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    total_tokens = sum(_estimate_tokens(t) for t in texts)
    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


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

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".wav")[1])
    try:
        tmp.write(await file.read())
        tmp.close()
        result = await processor.process(tmp.name, options={"language": language})
        return {"text": result.get("text", ""), "language": result.get("language", language)}
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Helpers for /v1/process/image
# ---------------------------------------------------------------------------

# Alias map so callers can use short names
_PROCESSOR_ALIASES: Dict[str, str] = {
    "ram": "ram",
    "florence": "florence2",
    "florence2": "florence2",
}
_ALL_IMAGE_PROCESSORS = ["ram", "florence2"]


def _resolve_processors(raw: str) -> List[str]:
    """
    Resolve a processors selector string into a list of internal model names.

    Accepted values:
      - ``"all"``      -> ["ram", "florence2"]
      - ``"ram"``      -> ["ram"]
      - ``"florence"``  -> ["florence2"]
      - comma-separated combination: ``"ram,florence"`` -> ["ram", "florence2"]
    """
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return list(_ALL_IMAGE_PROCESSORS)

    names: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        resolved = _PROCESSOR_ALIASES.get(token)
        if resolved is None:
            raise HTTPException(
                400,
                f"Unknown image processor '{token}'. "
                f"Valid values: {', '.join(sorted(_PROCESSOR_ALIASES.keys()))}, all",
            )
        if resolved not in names:
            names.append(resolved)
    return names or list(_ALL_IMAGE_PROCESSORS)


def _get_image_metadata(file_path: str) -> Dict[str, Any]:
    """Open *file_path* with PIL and return basic image metadata."""
    try:
        with PILImage.open(file_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": (img.format or "").upper() or None,
                "mode": img.mode,
            }
    except Exception as exc:
        logger.warning(f"Could not read image metadata from {file_path}: {exc}")
        return {}


def _merge_processor_results(
    per_processor: Dict[str, Dict[str, Any]],
    image_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge the raw results returned by individual processors into the unified
    response schema::

        {
          "tags": [{"name": "...", "confidence": 0.95, "category": "object"}],
          "descriptions": {"short": "...", "detailed": "..."},
          "metadata": {"width": 1024, "height": 768},
          "features": {"extracted_text": "..."},
          "processors_used": ["ram", "florence2"],
          "errors": { ... }          # only present when a processor failed
        }
    """
    merged_tags: Dict[str, Dict[str, Any]] = {}   # keyed by lowercase name for dedup
    merged_descriptions: Dict[str, str] = {}
    merged_features: Dict[str, Any] = {}
    processors_used: List[str] = []
    errors: Dict[str, str] = {}

    for proc_name, result in per_processor.items():
        if not result.get("success", False):
            errors[proc_name] = result.get("error", "unknown error")
            continue

        processors_used.append(proc_name)

        # --- tags ---
        for tag in result.get("tags", []):
            key = tag["name"].lower()
            existing = merged_tags.get(key)
            if existing is None or tag.get("confidence", 0) > existing.get("confidence", 0):
                merged_tags[key] = tag

        # --- descriptions ---
        for desc_key, desc_val in result.get("descriptions", {}).items():
            if desc_val:
                merged_descriptions[desc_key] = desc_val

        # --- features ---
        for feat_key, feat_val in result.get("features", {}).items():
            if feat_val is not None:
                merged_features[feat_key] = feat_val

        # --- metadata (processor-level, used as fallback) ---
        proc_meta = result.get("metadata", {})
        if proc_meta and not image_metadata:
            image_metadata = proc_meta

    # Sort tags by confidence descending
    sorted_tags = sorted(merged_tags.values(), key=lambda t: t.get("confidence", 0), reverse=True)

    response: Dict[str, Any] = {
        "tags": sorted_tags,
        "descriptions": merged_descriptions,
        "metadata": image_metadata,
        "features": merged_features,
        "processors_used": processors_used,
    }
    if errors:
        response["errors"] = errors
    return response


async def _run_processors(
    file_path: str,
    proc_names: List[str],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the requested image processors on *file_path* and return the unified
    merged response.
    """
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    image_metadata = _get_image_metadata(file_path)

    per_processor: Dict[str, Dict[str, Any]] = {}
    for proc_name in proc_names:
        processor = await _model_manager.get_or_load_model(proc_name, model_type="image")
        if processor:
            try:
                result = await processor.process(file_path, options)
                per_processor[proc_name] = result
            except Exception as e:
                logger.error(f"Image processor '{proc_name}' error: {e}", exc_info=True)
                per_processor[proc_name] = {"success": False, "error": str(e)}
        else:
            per_processor[proc_name] = {"success": False, "error": f"Could not load {proc_name}"}

    return _merge_processor_results(per_processor, image_metadata)


# ---------------------------------------------------------------------------
# Custom: /v1/process/image  (JSON body with file_path or base64)
# ---------------------------------------------------------------------------

@app.post("/v1/process/image")
async def process_image(req: ImageProcessRequest):
    """
    Unified image processing endpoint (JSON body).

    Accepts either a local ``file_path`` or ``image_base64``.  The
    ``processors`` field selects which models to run: ``"ram"``,
    ``"florence"`` (alias for florence2), ``"all"`` (default), or a
    comma-separated combination.

    Returns a merged result with tags, descriptions, metadata, features,
    and which processors were used.
    """
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    proc_names = _resolve_processors(req.processors)

    # Resolve image to a file path
    tmp_path: Optional[str] = None
    file_path = req.file_path

    if req.image_base64 and not file_path:
        try:
            image_data = base64.b64decode(req.image_base64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(image_data)
            tmp.close()
            file_path = tmp.name
            tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 image data: {e}")

    if not file_path:
        raise HTTPException(400, "Either 'file_path' or 'image_base64' must be provided")

    try:
        return await _run_processors(file_path, proc_names, req.options)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Custom: /v1/process/image/upload  (multipart file upload)
# ---------------------------------------------------------------------------

@app.post("/v1/process/image/upload")
async def process_image_upload(
    file: UploadFile = File(...),
    processors: str = Query("all", description='Processor selection: "ram", "florence", "all"'),
    options: str = Query("{}", description="JSON-encoded processing options"),
):
    """
    Image processing via file upload (multipart).

    ``processors`` accepts ``"ram"``, ``"florence"``, ``"all"`` (default),
    or a comma-separated combination.  Returns the same unified JSON
    structure as the JSON endpoint.
    """
    if not _model_manager:
        raise HTTPException(503, "Model manager not initialized")

    proc_names = _resolve_processors(processors)

    # Parse options JSON
    import json as _json
    try:
        opts = _json.loads(options) if options else {}
    except _json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in 'options' query parameter")

    suffix = os.path.splitext(file.filename or ".png")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        return await _run_processors(tmp.name, proc_names, opts)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
            loaded = model_data.get("loaded", False)
            # Include created timestamp for OpenAI compatibility
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(_start_time),
                "owned_by": "studiobrain",
                "type": model_type,
                "loaded": loaded,
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

    processor = await _model_manager.get_or_load_model(name)
    if processor:
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
