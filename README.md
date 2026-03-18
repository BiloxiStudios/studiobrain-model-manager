# StudioBrain Model Manager

Open-core GPU inference orchestrator with smart VRAM management. Runs multiple AI models (vision, audio, text, embeddings) on a single GPU with tier-based loading, automatic eviction, and OpenAI-compatible APIs.

## Features

- **Tiered VRAM management** — Tier 1 (always loaded), Tier 2 (on-demand), Tier 3 (exclusive, one at a time)
- **Process-level VRAM tracking** — Accurate in shared-GPU environments (multiple containers on one card)
- **Automatic eviction** — LRU unloading when VRAM pressure is high
- **OpenAI-compatible API** — Drop-in replacement for `/v1/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`
- **Custom image processing** — RAM tagging, Florence-2 captioning, BLIP-2 VQA in a single endpoint
- **YAML configuration** — Simple config file for models, tiers, and VRAM budgets

## Quick Start

```bash
pip install studiobrain-model-manager

# Copy and edit the example config
cp models.example.yaml models.yaml

# Start the server (default port 7070)
model-manager --config models.yaml
```

### Docker

```bash
docker build -t model-manager .
docker run --gpus all -p 7070:7070 model-manager
```

## API Endpoints

### OpenAI-Compatible

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/completions` | Text completion (routes to text/vision processors) |
| POST | `/v1/embeddings` | Text embeddings (sentence-transformers) |
| POST | `/v1/audio/transcriptions` | Audio transcription (Whisper) |
| GET | `/v1/models` | List available models + load status |

### Custom

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/process/image` | Multi-processor image analysis (RAM tags, Florence captions) |

### Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/models/status` | VRAM usage, loaded models, tier breakdown |
| POST | `/api/models/load/{name}` | Load a specific model |
| POST | `/api/models/unload/{name}` | Unload a specific model |
| POST | `/api/models/free-vram` | Force-unload all models |

## Configuration

See [models.example.yaml](models.example.yaml) for all options. Key settings:

```yaml
server:
  port: 7070

vram:
  budget_gb: 12  # Set in shared-GPU environments
  tier_1_limit_gb: 4.0
  tier_2_limit_gb: 8.0
  tier_3_limit_gb: 16.0

loading:
  preload_at_startup: true
  always_loaded: [ram, blip2]
  auto_unload_timeout_seconds: 300
```

Environment variable `VRAM_BUDGET_GB` overrides the YAML `budget_gb` setting.

## Architecture

```
Request → FastAPI Server → Model Manager → Registry → Processor
                              ↓
                        VRAM Monitor
                     (pynvml / PyTorch)
```

The **Model Manager** handles tier-based loading and VRAM pressure management. The **Registry** maps model names to processor classes and handles dynamic imports. The **VRAM Monitor** provides process-level GPU memory tracking that works correctly in shared-GPU environments.

## License

Apache 2.0
