"""
Auto-registration with LiteLLM proxy on startup.

When model-manager starts, it registers each available model with LiteLLM's
/model/new API so that LiteLLM can route inference requests to this instance.
On shutdown, it deregisters via /model/delete.

Configuration via environment variables:
    MODEL_MANAGER_HOSTNAME  - hostname/IP reachable from LiteLLM (default: auto-detect)
    LITELLM_REGISTRATION_URL - LiteLLM base URL (default: http://llm.braindead.games)
    LITELLM_API_KEY          - Bearer token for LiteLLM admin API
"""

import asyncio
import logging
import os
import socket
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LITELLM_BASE_URL = os.environ.get("LITELLM_REGISTRATION_URL", "http://llm.braindead.games")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
HOSTNAME = os.environ.get("MODEL_MANAGER_HOSTNAME", "")

# Retry settings
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10


def _get_hostname() -> str:
    """Return the hostname for this model-manager instance."""
    if HOSTNAME:
        return HOSTNAME
    return socket.gethostname()


def _build_model_name(model_id: str, hostname: str) -> str:
    """Build a LiteLLM model_name like 'ram@brainz'."""
    return f"{model_id}@{hostname}"


def _build_registration_payload(
    model_id: str,
    model_info: Dict,
    hostname: str,
    port: int,
) -> Dict:
    """Build the JSON payload for POST /model/new."""
    model_name = _build_model_name(model_id, hostname)
    description = model_info.get("description", f"{model_id} on {hostname}")
    capabilities = model_info.get("capabilities", [])
    model_type = model_info.get("type", "unknown")

    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/{model_id}",
            "api_base": f"http://{hostname}:{port}",
            "api_key": "local",
        },
        "model_info": {
            "description": f"{description} on {hostname.upper()} GPU",
            "capabilities": capabilities,
            "model_type": model_type,
        },
    }


def _collect_models_from_registry(registry) -> List[Dict]:
    """Extract flat list of {id, info} dicts from the ModelRegistry."""
    models = []
    for model_type, model_dict in registry.available_models.items():
        for model_id, model_data in model_dict.items():
            models.append({
                "id": model_id,
                "info": {**model_data, "type": model_type},
            })
    return models


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

async def register_with_litellm(registry, port: int) -> None:
    """
    Register all models from the registry with LiteLLM.

    Runs in the background with retries so it doesn't block startup.
    """
    if not LITELLM_API_KEY:
        logger.warning("LITELLM_API_KEY not set -- skipping LiteLLM registration")
        return

    asyncio.create_task(_register_with_retries(registry, port))


async def _register_with_retries(registry, port: int) -> None:
    """Retry loop for LiteLLM registration."""
    hostname = _get_hostname()
    models = _collect_models_from_registry(registry)

    if not models:
        logger.warning("No models found in registry -- nothing to register")
        return

    logger.info(
        "LiteLLM registration: %d models to register at %s (hostname=%s, port=%d)",
        len(models), LITELLM_BASE_URL, hostname, port,
    )

    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                registered = 0
                failed = 0

                for model in models:
                    payload = _build_registration_payload(
                        model["id"], model["info"], hostname, port,
                    )
                    try:
                        resp = await client.post(
                            f"{LITELLM_BASE_URL}/model/new",
                            json=payload,
                            headers=headers,
                        )
                        if resp.status_code in (200, 201):
                            registered += 1
                            logger.debug(
                                "Registered %s with LiteLLM", payload["model_name"]
                            )
                        else:
                            failed += 1
                            logger.warning(
                                "Failed to register %s: %d %s",
                                payload["model_name"], resp.status_code, resp.text[:200],
                            )
                    except Exception as e:
                        failed += 1
                        logger.warning(
                            "Error registering %s: %s", payload["model_name"], e,
                        )

                logger.info(
                    "LiteLLM registration complete: %d registered, %d failed",
                    registered, failed,
                )
                return  # Success -- exit retry loop

        except Exception as e:
            logger.warning(
                "LiteLLM registration attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, e,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECONDS)

    logger.error(
        "LiteLLM registration failed after %d attempts -- models NOT registered",
        MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Deregistration
# ---------------------------------------------------------------------------

async def deregister_from_litellm(registry, port: int) -> None:
    """
    Deregister all models from LiteLLM on shutdown.

    Best-effort -- failures are logged but don't block shutdown.
    """
    if not LITELLM_API_KEY:
        return

    hostname = _get_hostname()
    models = _collect_models_from_registry(registry)

    if not models:
        return

    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("Deregistering %d models from LiteLLM...", len(models))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for model in models:
                model_name = _build_model_name(model["id"], hostname)
                try:
                    resp = await client.post(
                        f"{LITELLM_BASE_URL}/model/delete",
                        json={"id": model_name},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        logger.debug("Deregistered %s from LiteLLM", model_name)
                    else:
                        logger.debug(
                            "Deregister %s: %d (may not have been registered)",
                            model_name, resp.status_code,
                        )
                except Exception as e:
                    logger.debug("Error deregistering %s: %s", model_name, e)

        logger.info("LiteLLM deregistration complete")
    except Exception as e:
        logger.warning("LiteLLM deregistration error: %s", e)
