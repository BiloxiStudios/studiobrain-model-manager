"""
VRAM Monitor — GPU memory tracking for shared and dedicated environments.

Extracted from model_manager.py. Provides process-level and system-level VRAM
measurement using pynvml (preferred) with PyTorch fallback.

In shared-GPU environments (multiple containers on one GPU), device-level pynvml
queries report combined VRAM of ALL processes. This module uses
nvmlDeviceGetComputeRunningProcesses() filtered to the current PID to give an
accurate per-process figure.
"""

import logging
import os
from typing import Tuple

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """
    Monitors GPU VRAM usage at both process and system level.

    Args:
        vram_budget_gb: Explicit VRAM ceiling for this process (0 = use physical GPU total).
    """

    def __init__(self, vram_budget_gb: float = 0.0):
        self.vram_budget_gb = vram_budget_gb
        self._physical_gpu_index: int | None = None

    @property
    def budget_gb(self) -> float:
        """Effective VRAM budget for this process."""
        if self.vram_budget_gb > 0:
            return self.vram_budget_gb
        _, total = self.get_system_vram_usage()
        return total

    # ------------------------------------------------------------------
    # Physical GPU detection
    # ------------------------------------------------------------------

    def get_physical_gpu_index(self) -> int:
        """
        Find the physical GPU index that PyTorch is using.
        Maps PyTorch's logical device 0 to the physical GPU index for pynvml.
        """
        if self._physical_gpu_index is not None:
            return self._physical_gpu_index

        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            return 0

        try:
            pytorch_gpu_name = torch.cuda.get_device_name(0)
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for physical_idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
                physical_gpu_name = pynvml.nvmlDeviceGetName(handle)

                if isinstance(physical_gpu_name, bytes):
                    physical_gpu_name = physical_gpu_name.decode("utf-8")

                if pytorch_gpu_name in physical_gpu_name or physical_gpu_name in pytorch_gpu_name:
                    pynvml.nvmlShutdown()
                    logger.info(f"Mapped PyTorch device 0 ({pytorch_gpu_name}) to physical GPU {physical_idx}")
                    self._physical_gpu_index = physical_idx
                    return physical_idx

            pynvml.nvmlShutdown()
            logger.warning(f"Could not find physical GPU matching '{pytorch_gpu_name}', defaulting to GPU 0")
            return 0

        except Exception as e:
            logger.warning(f"Failed to detect physical GPU index: {e}, defaulting to GPU 0")
            return 0

    # ------------------------------------------------------------------
    # System-wide VRAM
    # ------------------------------------------------------------------

    def get_system_vram_usage(self) -> Tuple[float, float]:
        """
        Get system-wide VRAM usage (includes all processes).

        Returns:
            (used_gb, total_gb) for the GPU.
        """
        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            return (0.0, 24.0)

        try:
            physical_gpu_idx = self.get_physical_gpu_index()
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()

            used_gb = mem_info.used / (1024 ** 3)
            total_gb = mem_info.total / (1024 ** 3)
            return (used_gb, total_gb)
        except Exception as e:
            logger.warning(f"Failed to get system VRAM via pynvml: {e}")
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return (0.0, total_gb)
            except Exception:
                return (0.0, 24.0)

    # ------------------------------------------------------------------
    # Process-level VRAM
    # ------------------------------------------------------------------

    def get_process_vram_usage(self) -> float:
        """
        Get VRAM used by THIS process only, using pynvml process-level queries.
        Falls back to PyTorch memory_reserved() when pynvml is unavailable.

        Returns:
            VRAM used by this process in GB.
        """
        if not torch.cuda.is_available():
            return 0.0

        our_pid = os.getpid()

        if PYNVML_AVAILABLE:
            try:
                physical_gpu_idx = self.get_physical_gpu_index()
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_idx)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                pynvml.nvmlShutdown()

                for proc in procs:
                    if proc.pid == our_pid:
                        used_gb = proc.usedGpuMemory / (1024 ** 3)
                        logger.debug(f"Process-level VRAM (pynvml): {used_gb:.2f} GB for PID {our_pid}")
                        return used_gb

                logger.debug(f"PID {our_pid} not in GPU process list -- returning 0 GB")
                return 0.0

            except Exception as e:
                logger.warning(
                    f"pynvml process-level VRAM query failed: {e}. "
                    f"Falling back to PyTorch memory_reserved()."
                )

        # Fallback: PyTorch reserved memory (this process only)
        try:
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            logger.debug(f"Process-level VRAM (PyTorch reserved): {reserved:.2f} GB")
            return reserved
        except Exception:
            return 0.0

    def get_app_vram_usage(self) -> float:
        """
        Get VRAM usage for THIS app only (PyTorch reserved memory).
        Does not include other processes.

        Returns:
            App VRAM usage in GB.
        """
        if not torch.cuda.is_available():
            return 0.0
        try:
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return reserved
        except Exception:
            return 0.0
