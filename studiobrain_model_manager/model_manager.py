"""
Dynamic Model Manager Service
Manages model loading/unloading with intelligent VRAM monitoring and tiered loading strategy.

VRAM tracking in shared GPU environments
-----------------------------------------
On machines where multiple containers share a single GPU (e.g. qwen32b-vl running Qwen 32B
via vLLM plus localai alongside this service), a device-level pynvml query reports the sum
of ALL process VRAM (potentially >> the physical card size in paged scenarios, or simply
very high usage from co-tenants). This makes `available_gb = max(0, budget - system_used)`
collapse to 0 and prevents any model from loading.

Fix (implemented below):
  1. Process-level tracking (default): use pynvml.nvmlDeviceGetComputeRunningProcesses()
     filtered to this process's PID to measure only studiobrain-ai's own VRAM footprint.
     Falls back to PyTorch memory_reserved() when pynvml is unavailable.
  2. VRAM_BUDGET_GB override: if operators set VRAM_BUDGET_GB, `_get_current_vram_usage()`
     returns this process's own usage (not system-wide), and the budget ceiling is respected.
     This gives deterministic, explicit control in shared-GPU deployments.

The legacy `_get_system_vram_usage()` is retained for status/reporting only.
"""

import logging
import asyncio
import os
import time
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from pathlib import Path
import torch
import psutil
from dataclasses import dataclass

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.debug("pynvml not available - system VRAM monitoring will be limited (install nvidia-ml-py to enable)")

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """Model loading tiers based on VRAM usage and loading strategy"""
    TIER_1_ALWAYS = "tier1_always"    # Always loaded (BLIP2, RAM) - 2-3GB
    TIER_2_ON_DEMAND = "tier2_on_demand"  # Load on demand (Florence2, SD) - 3-4GB
    TIER_3_EXCLUSIVE = "tier3_exclusive"  # Exclusive loading (Qwen) - 7GB+

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    tier: ModelTier
    processor: Any
    vram_usage: float  # GB
    last_used: float  # Timestamp
    load_time: float  # Seconds to load
    use_count: int = 0

class ModelManager:
    """
    Dynamic Model Manager with intelligent VRAM monitoring

    Loading Strategy:
    - Tier 1 (Always Loaded): BLIP2, RAM for asset system (~3GB total)
    - Tier 2 (On-Demand): Florence2, Stable Diffusion (~6GB budget)
    - Tier 3 (Exclusive): Qwen models (~13GB budget, only one at a time)
    """

    def __init__(self, settings, model_registry):
        self.settings = settings
        self.model_registry = model_registry
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_cache_timeout = settings.model_cache_timeout
        self.max_vram_usage = settings.max_vram_usage

        # vram_budget_gb > 0 means the operator has set an explicit VRAM ceiling for this
        # process in a shared-GPU environment.  When active, we track only our own process
        # VRAM usage (not system-wide) and compare against this budget instead of
        # max_vram_usage.  When 0 (default), behaviour is unchanged: we still use
        # process-level VRAM but compare against max_vram_usage.
        self.vram_budget_gb: float = getattr(settings, "vram_budget_gb", 0.0)
        if self.vram_budget_gb > 0:
            logger.info(
                f"VRAM_BUDGET_GB={self.vram_budget_gb:.1f} — shared-GPU mode active. "
                f"VRAM accounting will use process-level usage only."
            )
            # Override max_vram_usage with the explicit budget so all downstream
            # comparisons (tier checks, pressure checks) respect the operator setting.
            self.max_vram_usage = self.vram_budget_gb

        # Model tier configuration
        self.model_tiers = {
            # Tier 1: Always loaded (lightweight vision models)
            "blip2": ModelTier.TIER_1_ALWAYS,
            "ram": ModelTier.TIER_1_ALWAYS,

            # Tier 2: On-demand (moderate VRAM)
            "florence2": ModelTier.TIER_2_ON_DEMAND,
            "stable_diffusion": ModelTier.TIER_2_ON_DEMAND,

            # Tier 3: Exclusive (heavy VRAM)
            "qwen_image": ModelTier.TIER_3_EXCLUSIVE,
            "qwen_text": ModelTier.TIER_3_EXCLUSIVE,
            "qwen_image_edit": ModelTier.TIER_3_EXCLUSIVE
        }

        # VRAM limits per tier
        self.tier_vram_limits = {
            ModelTier.TIER_1_ALWAYS: settings.tier_1_vram_limit,
            ModelTier.TIER_2_ON_DEMAND: settings.tier_2_vram_limit,
            ModelTier.TIER_3_EXCLUSIVE: settings.tier_3_vram_limit
        }

        # Estimated VRAM usage per model (GB)
        self.model_vram_estimates = {
            "blip2": 1.5,
            "ram": 1.2,
            "florence2": 3.0,
            "stable_diffusion": 4.0,
            "qwen_image": 7.0,
            "qwen_text": 16.0,
            "qwen_image_edit": 6.0,
            "qwen3_vl": 17.0,
            "qwen2_vl": 16.0
        }

        self._cleanup_task = None

    async def initialize(self):
        """Initialize the model manager and load Tier 1 models"""
        logger.info("Initializing Dynamic Model Manager...")

        # Log VRAM configuration — show both system-wide and process-local figures
        # so operators can diagnose shared-GPU environments at a glance.
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            system_used, _ = self._get_system_vram_usage()
            process_used = self._get_process_vram_usage()
            other_used = max(0.0, system_used - process_used)
            logger.info(
                f"GPU VRAM: {total_vram:.1f}GB physical | "
                f"system-wide used: {system_used:.1f}GB | "
                f"this process: {process_used:.1f}GB | "
                f"other processes: {other_used:.1f}GB | "
                f"budget for this service: {self.max_vram_usage}GB"
            )
            if other_used > 1.0:
                logger.warning(
                    f"Shared GPU detected: {other_used:.1f}GB consumed by other processes. "
                    f"VRAM tracking uses process-level measurement only. "
                    f"Set VRAM_BUDGET_GB to cap this service's allocation explicitly."
                )

        # Load Tier 1 models if preload is enabled
        if self.settings.preload_models_at_startup:
            logger.info("Preloading Tier 1 models at startup...")
            for model_name in self.settings.always_loaded_models:
                if model_name in self.model_tiers:
                    success = await self._load_model_internal(model_name, force=True)
                    if not success:
                        logger.warning(f"Failed to load always-loaded model: {model_name}")
        else:
            logger.info("Model preload disabled - models will load on-demand")

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(f"Model Manager initialized. Loaded models: {list(self.loaded_models.keys())}")

    async def get_or_load_model(self, model_name: str, model_type: str = None) -> Optional[Any]:
        """
        Get a model processor, loading it if necessary

        Args:
            model_name: Name of the model to load
            model_type: Type hint for registry lookup

        Returns:
            Model processor or None if failed to load
        """
        # Check if already loaded
        if model_name in self.loaded_models:
            model_info = self.loaded_models[model_name]
            model_info.last_used = time.time()
            model_info.use_count += 1
            logger.debug(f"Using cached model: {model_name}")
            return model_info.processor

        # Load the model
        success = await self._load_model_internal(model_name)
        if success:
            return self.loaded_models[model_name].processor

        return None

    async def unload_model(self, model_name: str, force: bool = False) -> bool:
        """
        Unload a specific model from memory

        Args:
            model_name: Name of model to unload
            force: Force unload even if it's a Tier 1 model

        Returns:
            True if successfully unloaded
        """
        if model_name not in self.loaded_models:
            return True

        model_info = self.loaded_models[model_name]

        # Don't unload Tier 1 models unless forced
        if model_info.tier == ModelTier.TIER_1_ALWAYS and not force:
            logger.debug(f"Skipping unload of Tier 1 model: {model_name}")
            return False

        try:
            # Call model cleanup if available
            if hasattr(model_info.processor, 'cleanup'):
                await model_info.processor.cleanup()

            # Remove from registry
            await self.model_registry.unload_model("auto", model_name)  # Let registry figure out type

            # Explicitly break references to help garbage collection
            if hasattr(model_info, 'processor'):
                processor = model_info.processor
                # If processor has models/pipelines, delete them explicitly
                if hasattr(processor, 'model'):
                    del processor.model
                if hasattr(processor, 'pipeline'):
                    del processor.pipeline
                if hasattr(processor, 'processor'):
                    del processor.processor
                # Delete the processor reference
                model_info.processor = None
                del processor

            # Remove from our tracking
            del self.loaded_models[model_name]

            # Explicitly set model_info to None to break reference
            vram_usage = model_info.vram_usage  # Save for logging
            model_info = None

            # Force aggressive GPU memory cleanup
            import gc
            gc.collect()  # Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # PyTorch CUDA cache
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                torch.cuda.ipc_collect()  # Collect inter-process CUDA memory

            # Additional aggressive cleanup
            gc.collect()  # Run GC again after CUDA cleanup

            logger.info(f"Unloaded model: {model_name} (freed ~{vram_usage:.1f}GB VRAM)")
            return True

        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}", exc_info=True)
            return False

    async def _load_model_internal(self, model_name: str, force: bool = False) -> bool:
        """
        Internal method to load a model with VRAM management

        Args:
            model_name: Name of model to load
            force: Force load even if VRAM is insufficient

        Returns:
            True if successfully loaded
        """
        if model_name in self.loaded_models:
            return True

        if model_name not in self.model_tiers:
            logger.error(f"Unknown model tier for: {model_name}")
            return False

        tier = self.model_tiers[model_name]
        estimated_vram = self.model_vram_estimates.get(model_name, 5.0)

        # Check VRAM availability
        current_vram = await self._get_current_vram_usage()
        available_vram = self.max_vram_usage - current_vram

        if not force and estimated_vram > available_vram:
            # Try to free up space
            if tier == ModelTier.TIER_3_EXCLUSIVE:
                # Unload other Tier 3 models
                await self._unload_tier_models(ModelTier.TIER_3_EXCLUSIVE)
                current_vram = await self._get_current_vram_usage()
                available_vram = self.max_vram_usage - current_vram

            if estimated_vram > available_vram:
                # Try unloading least recently used Tier 2 models
                await self._unload_lru_models(estimated_vram - available_vram)
                current_vram = await self._get_current_vram_usage()
                available_vram = self.max_vram_usage - current_vram

            if estimated_vram > available_vram:
                logger.error(f"Insufficient VRAM to load {model_name}: need {estimated_vram:.1f}GB, have {available_vram:.1f}GB")
                return False

        # Determine model type for registry
        model_type = self._get_model_type(model_name)
        if not model_type:
            logger.error(f"Unknown model type for: {model_name}")
            return False

        # Load the model
        start_time = time.time()
        logger.info(f"Loading {tier.value} model: {model_name} (estimated {estimated_vram:.1f}GB VRAM)")

        success = await self.model_registry.load_model(model_type, model_name)

        if success:
            load_time = time.time() - start_time
            processor = self.model_registry.get_processor(model_type, model_name)

            # Track the loaded model
            self.loaded_models[model_name] = ModelInfo(
                name=model_name,
                tier=tier,
                processor=processor,
                vram_usage=estimated_vram,  # TODO: Get actual VRAM usage
                last_used=time.time(),
                load_time=load_time
            )

            logger.info(f"Successfully loaded {model_name} in {load_time:.1f}s")
            return True
        else:
            logger.error(f"Failed to load model: {model_name}")
            return False

    def _get_model_type(self, model_name: str) -> Optional[str]:
        """Get the model type for registry lookup"""
        type_mapping = {
            "blip2": "image",
            "ram": "image",
            "florence2": "image",
            "stable_diffusion": "image",
            "qwen_image": "image",
            "qwen_image_edit": "image_edit",
            "qwen_text": "text"
        }
        return type_mapping.get(model_name)

    def _get_physical_gpu_index(self) -> int:
        """
        Find the physical GPU index that PyTorch is using.
        This maps PyTorch's logical device 0 to the physical GPU index for pynvml.

        Returns:
            Physical GPU index (for pynvml), or 0 if detection fails
        """
        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            return 0

        try:
            # Get GPU name from PyTorch (e.g., "NVIDIA GeForce RTX 4090")
            pytorch_gpu_name = torch.cuda.get_device_name(0)

            # Query all physical GPUs to find matching name
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for physical_idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
                physical_gpu_name = pynvml.nvmlDeviceGetName(handle)

                # Match GPU names (decode bytes if needed)
                if isinstance(physical_gpu_name, bytes):
                    physical_gpu_name = physical_gpu_name.decode('utf-8')

                if pytorch_gpu_name in physical_gpu_name or physical_gpu_name in pytorch_gpu_name:
                    pynvml.nvmlShutdown()
                    logger.info(f"Mapped PyTorch device 0 ({pytorch_gpu_name}) to physical GPU {physical_idx}")
                    return physical_idx

            pynvml.nvmlShutdown()
            logger.warning(f"Could not find physical GPU matching '{pytorch_gpu_name}', defaulting to GPU 0")
            return 0

        except Exception as e:
            logger.warning(f"Failed to detect physical GPU index: {e}, defaulting to GPU 0")
            return 0

    def _get_system_vram_usage(self) -> tuple[float, float]:
        """
        Get system-wide VRAM usage using pynvml (includes all processes like ComfyUI).
        Automatically detects which physical GPU PyTorch is using.

        Returns:
            Tuple of (used_gb, total_gb) for the GPU
        """
        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            return (0.0, 24.0)  # Fallback defaults

        try:
            # Get the physical GPU index that matches PyTorch's device 0
            physical_gpu_idx = self._get_physical_gpu_index()

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()

            used_gb = mem_info.used / (1024**3)
            total_gb = mem_info.total / (1024**3)
            return (used_gb, total_gb)
        except Exception as e:
            logger.warning(f"Failed to get system VRAM via pynvml: {e}")
            # Fallback to PyTorch
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return (0.0, total_gb)
            except:
                return (0.0, 24.0)

    def _get_app_vram_usage(self) -> float:
        """
        Get VRAM usage for THIS app only (PyTorch reserved memory).
        Does not include other processes like ComfyUI.

        Returns:
            App VRAM usage in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        try:
            # Use memory_reserved to include cached memory not yet freed
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            return reserved
        except Exception:
            # Fallback: sum estimated usage of loaded models
            return sum(model.vram_usage for model in self.loaded_models.values())

    def _get_process_vram_usage(self) -> float:
        """
        Get VRAM used by THIS process only, using pynvml process-level queries.

        In shared-GPU environments other containers (e.g. qwen32b-vl, localai) can
        consume the majority of system VRAM. Using device-level pynvml queries would
        report the combined usage of ALL processes, making available_gb = 0.

        This method uses nvmlDeviceGetComputeRunningProcesses() to find only the
        entry matching our own PID, giving an accurate per-process figure.

        Falls back to PyTorch memory_reserved() when pynvml is unavailable, which
        is also process-local (PyTorch only tracks its own allocations).

        Returns:
            VRAM used by this process in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        our_pid = os.getpid()

        if PYNVML_AVAILABLE:
            try:
                physical_gpu_idx = self._get_physical_gpu_index()
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_idx)

                # nvmlDeviceGetComputeRunningProcesses returns a list of
                # nvmlProcessInfo_t objects with fields: pid and usedGpuMemory.
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                pynvml.nvmlShutdown()

                for proc in procs:
                    if proc.pid == our_pid:
                        used_gb = proc.usedGpuMemory / (1024**3)
                        logger.debug(
                            f"Process-level VRAM (pynvml): {used_gb:.2f} GB for PID {our_pid}"
                        )
                        return used_gb

                # Our PID not found — no GPU memory allocated yet
                logger.debug(f"PID {our_pid} not in GPU process list — returning 0 GB")
                return 0.0

            except Exception as e:
                logger.warning(
                    f"pynvml process-level VRAM query failed: {e}. "
                    f"Falling back to PyTorch memory_reserved()."
                )

        # Fallback: PyTorch reserved memory (this process only)
        try:
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.debug(f"Process-level VRAM (PyTorch reserved): {reserved:.2f} GB")
            return reserved
        except Exception:
            # Last resort: sum of tracked model estimates
            return sum(model.vram_usage for model in self.loaded_models.values())

    async def _get_current_vram_usage(self) -> float:
        """
        Get current VRAM usage in GB for THIS process only.

        Changed from system-wide measurement to process-level measurement to support
        shared-GPU deployments where co-located containers (e.g. qwen32b-vl, localai)
        consume most of system VRAM and would cause available_gb to collapse to 0.

        When VRAM_BUDGET_GB is set in the environment (vram_budget_gb > 0 on settings),
        this value is compared against that budget ceiling instead of max_vram_usage.
        See __init__ where max_vram_usage is overridden with vram_budget_gb.

        For system-wide VRAM reporting (status endpoints), use _get_system_vram_usage().
        """
        return self._get_process_vram_usage()

    async def check_and_free_vram_if_needed(self, vram_needed: float = 0) -> Dict[str, Any]:
        """
        Check if VRAM pressure is high and free memory if needed.
        Called before loading new models, periodically, or by external requests (ComfyUI).

        Args:
            vram_needed: Additional VRAM needed in GB (for pre-emptive freeing)

        Returns:
            Dict with freed status, amount freed, and current usage
        """
        current_vram = await self._get_current_vram_usage()

        # Use max_vram_usage (our service budget) as the ceiling for pressure calculations.
        # This is either the VRAM_BUDGET_GB value (in shared-GPU environments) or
        # the MAX_VRAM_USAGE setting.  We no longer use the physical GPU total here
        # because current_vram is now process-level (not system-wide), and comparing
        # our process's usage against the physical card total would always look fine
        # even when we're near our intended budget limit.
        budget_vram = self.max_vram_usage
        # Alias kept for readability inside the if block below (thresholds like * 0.85)
        total_vram = budget_vram

        # Calculate pressure (how full is our budget?)
        vram_usage_percent = (current_vram / budget_vram) * 100 if budget_vram > 0 else 0.0

        # Determine if we're in critical state
        # Critical if: >90% of our budget used OR not enough space for requested amount
        critical = vram_usage_percent > 90 or (current_vram + vram_needed > budget_vram * 0.9)

        if critical:
            logger.warning(
                f"[VRAM PRESSURE] HIGH: {current_vram:.1f}GB/{budget_vram:.1f}GB ({vram_usage_percent:.0f}% of budget) "
                f"- Need {vram_needed:.1f}GB more"
            )

            # Unload models in priority order (Tier 3 first, then Tier 2, then Tier 1 if needed)
            models_unloaded = []

            # Tier 3 models (heavy, exclusive)
            tier3_models = [
                name for name, model in self.loaded_models.items()
                if model.tier == ModelTier.TIER_3_EXCLUSIVE
            ]
            for model_name in tier3_models:
                await self.unload_model(model_name, force=True)
                models_unloaded.append(model_name)
                logger.info(f"[PRESSURE UNLOAD] Tier 3: {model_name}")

            # Re-check after Tier 3
            current_vram = await self._get_current_vram_usage()
            if current_vram + vram_needed <= total_vram * 0.85:
                return self._create_pressure_result(True, models_unloaded, current_vram)

            # Tier 2 models (on-demand)
            tier2_models = [
                name for name, model in self.loaded_models.items()
                if model.tier == ModelTier.TIER_2_ON_DEMAND
            ]
            for model_name in tier2_models:
                await self.unload_model(model_name, force=True)
                models_unloaded.append(model_name)
                logger.info(f"[PRESSURE UNLOAD] Tier 2: {model_name}")

            # Re-check after Tier 2
            current_vram = await self._get_current_vram_usage()
            if current_vram + vram_needed <= total_vram * 0.85:
                return self._create_pressure_result(True, models_unloaded, current_vram)

            # Tier 1 models (always loaded, only if tier1_always_loaded=False OR critical)
            if not self.settings.tier1_models_always_loaded or vram_usage_percent > 95:
                tier1_models = [
                    name for name, model in self.loaded_models.items()
                    if model.tier == ModelTier.TIER_1_ALWAYS
                ]
                for model_name in tier1_models:
                    await self.unload_model(model_name, force=True)
                    models_unloaded.append(model_name)
                    logger.warning(f"[PRESSURE UNLOAD] Tier 1 (critical): {model_name}")

            # Aggressive cleanup after pressure unload
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

            # Final VRAM check
            final_vram = await self._get_current_vram_usage()
            freed = current_vram - final_vram
            logger.info(f"[PRESSURE UNLOAD] Complete: Freed {freed:.1f}GB, {len(models_unloaded)} models unloaded")

            return self._create_pressure_result(True, models_unloaded, final_vram, freed)

        # No pressure, return status
        return {
            "pressure_detected": False,
            "models_unloaded": [],
            "current_vram_gb": round(current_vram, 2),
            "vram_freed_gb": 0.0
        }

    def _create_pressure_result(self, pressure: bool, models: List[str], current_vram: float, freed: float = 0) -> Dict[str, Any]:
        """Helper to create pressure unload result dict"""
        return {
            "pressure_detected": pressure,
            "models_unloaded": models,
            "current_vram_gb": round(current_vram, 2),
            "vram_freed_gb": round(freed, 2)
        }

    async def _unload_tier_models(self, tier: ModelTier):
        """Unload all models of a specific tier"""
        models_to_unload = [
            name for name, model in self.loaded_models.items()
            if model.tier == tier
        ]

        for model_name in models_to_unload:
            await self.unload_model(model_name, force=True)

    async def _unload_lru_models(self, vram_needed: float):
        """Unload least recently used models to free up VRAM"""
        # Sort by last used time (oldest first), excluding Tier 1
        lru_models = [
            (name, model) for name, model in self.loaded_models.items()
            if model.tier != ModelTier.TIER_1_ALWAYS
        ]
        lru_models.sort(key=lambda x: x[1].last_used)

        vram_freed = 0.0
        for model_name, model_info in lru_models:
            if vram_freed >= vram_needed:
                break

            success = await self.unload_model(model_name, force=True)
            if success:
                vram_freed += model_info.vram_usage
                logger.info(f"LRU unloaded {model_name} (freed {model_info.vram_usage:.1f}GB)")

    async def _cleanup_loop(self):
        """Background task to clean up unused models"""
        while True:
            try:
                current_time = time.time()
                models_to_unload = []

                # Use configurable timeout
                timeout = self.settings.model_auto_unload_timeout

                for name, model in self.loaded_models.items():
                    # Skip Tier 1 models if they should always be loaded
                    if model.tier == ModelTier.TIER_1_ALWAYS and self.settings.tier1_models_always_loaded:
                        continue

                    # Check if model hasn't been used recently
                    if current_time - model.last_used > timeout:
                        models_to_unload.append(name)

                # Unload expired models
                for model_name in models_to_unload:
                    await self.unload_model(model_name, force=True)
                    logger.info(f"Cache expired: unloaded {model_name} after {timeout}s inactivity")

                # Force aggressive garbage collection and VRAM clearing after timeout unloads
                if len(models_to_unload) > 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all operations to complete
                    logger.info(f"Forced VRAM cleanup after auto-unload of {len(models_to_unload)} models")

                # Also check for VRAM pressure (for ComfyUI compatibility)
                # This will unload models if VRAM usage is >90% even if timeout hasn't occurred
                pressure_result = await self.check_and_free_vram_if_needed(vram_needed=0)
                if pressure_result.get("pressure_detected", False):
                    logger.info(
                        f"Pressure monitoring: Freed {pressure_result['vram_freed_gb']}GB, "
                        f"unloaded {len(pressure_result['models_unloaded'])} models"
                    )

                # Wait before next cleanup cycle
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model cleanup loop: {e}")
                await asyncio.sleep(60)

    async def force_unload_all(self) -> Dict[str, Any]:
        """
        Force unload ALL models from GPU memory (including Tier 1)

        Returns:
            Dictionary with unload results and VRAM freed
        """
        logger.info("Force unloading all models from GPU...")

        # Get current VRAM before unload
        vram_before = await self._get_current_vram_usage()

        models_unloaded = []
        models_failed = []

        # Unload all models with force=True
        for model_name in list(self.loaded_models.keys()):
            success = await self.unload_model(model_name, force=True)
            if success:
                models_unloaded.append(model_name)
            else:
                models_failed.append(model_name)

        # Get VRAM after unload
        vram_after = await self._get_current_vram_usage()
        vram_freed = max(0, vram_before - vram_after)

        result = {
            "success": len(models_failed) == 0,
            "models_unloaded": models_unloaded,
            "models_failed": models_failed,
            "vram_freed_gb": round(vram_freed, 2),
            "vram_before_gb": round(vram_before, 2),
            "vram_after_gb": round(vram_after, 2)
        }

        logger.info(f"Force unload complete: {len(models_unloaded)} models unloaded, {vram_freed:.2f}GB freed")
        return result

    async def get_loaded_models_status(self) -> Dict[str, Any]:
        """
        Get detailed status of currently loaded models with VRAM info

        Returns:
            Detailed status including loaded models, VRAM usage (both app and system), and config settings
        """
        # Get system-wide VRAM usage (includes ComfyUI and other processes)
        system_used_gb, total_vram_gb = self._get_system_vram_usage()

        # Get app-only VRAM usage (PyTorch reserved memory for this process)
        app_used_gb = self._get_app_vram_usage()

        return {
            "loaded_models": [
                {
                    "name": name,
                    "tier": model.tier.value,
                    "vram_usage_gb": round(model.vram_usage, 2),
                    "last_used_seconds_ago": int(time.time() - model.last_used),
                    "use_count": model.use_count,
                    "load_time_seconds": round(model.load_time, 2)
                }
                for name, model in self.loaded_models.items()
            ],
            "vram": {
                # System-wide VRAM (all processes including ComfyUI)
                "current_usage_gb": round(system_used_gb, 2),
                "total_gpu_vram_gb": round(total_vram_gb, 2),
                "available_gb": round(max(0, total_vram_gb - system_used_gb), 2),

                # App-only VRAM (this AI service)
                "app_usage_gb": round(app_used_gb, 2),
                "app_available_gb": round(max(0, self.max_vram_usage - app_used_gb), 2),
                "max_usage_gb": self.max_vram_usage,

                # Other processes VRAM (ComfyUI, etc)
                "other_processes_gb": round(max(0, system_used_gb - app_used_gb), 2)
            },
            "settings": {
                "preload_at_startup": self.settings.preload_models_at_startup,
                "tier1_always_loaded": self.settings.tier1_models_always_loaded,
                "auto_unload_timeout_seconds": self.settings.model_auto_unload_timeout
            },
            "model_count": {
                "total": len(self.loaded_models),
                "tier1": sum(1 for m in self.loaded_models.values() if m.tier == ModelTier.TIER_1_ALWAYS),
                "tier2": sum(1 for m in self.loaded_models.values() if m.tier == ModelTier.TIER_2_ON_DEMAND),
                "tier3": sum(1 for m in self.loaded_models.values() if m.tier == ModelTier.TIER_3_EXCLUSIVE)
            }
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current model manager status (legacy method, use get_loaded_models_status)"""
        current_vram = await self._get_current_vram_usage()

        return {
            "loaded_models": {
                name: {
                    "tier": model.tier.value,
                    "vram_usage_gb": model.vram_usage,
                    "last_used": model.last_used,
                    "use_count": model.use_count,
                    "load_time": model.load_time
                }
                for name, model in self.loaded_models.items()
            },
            "vram": {
                "current_usage_gb": current_vram,
                "max_usage_gb": self.max_vram_usage,
                "available_gb": max(0, self.max_vram_usage - current_vram)
            },
            "tier_limits": {
                tier.value: limit for tier, limit in self.tier_vram_limits.items()
            }
        }

    async def cleanup(self):
        """Clean up the model manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name, force=True)

        logger.info("Model Manager cleanup completed")