FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml README.md ./
RUN pip3 install --no-cache-dir .

# Copy source
COPY studiobrain_model_manager/ ./studiobrain_model_manager/
COPY models.example.yaml ./models.yaml

# Default port
EXPOSE 7070

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7070/health || exit 1

ENTRYPOINT ["model-manager"]
CMD ["--config", "models.yaml"]
