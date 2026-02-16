"""
Platform-aware device selection for LLM inference.

- **PC (Windows/Linux) with NVIDIA GPU (e.g. RTX 3080):** uses `cuda` for GPU acceleration.
  Install with: `pip install "gpt4all[cuda]"` and ensure NVIDIA drivers + CUDA are installed.

- **MacBook (macOS on Apple Silicon, e.g. M4):** uses `gpu` (Metal) when available.
  Standard `pip install gpt4all` is sufficient.

Override via environment variable: set `GPT4ALL_DEVICE` to `cuda`, `gpu`, or `cpu`.
"""

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)


def get_llm_device() -> str:
    """
    Choose the GPT4All device string for the current platform.

    - Windows/Linux: prefer `cuda` for NVIDIA GPU (e.g. RTX 3080).
    - macOS on Apple Silicon: use `gpu` (Metal).
    - Otherwise: `cpu`.

    Can be overridden with the environment variable `GPT4ALL_DEVICE`
    (e.g. `cuda`, `gpu`, `cpu`).

    Returns:
        One of: "cuda", "gpu", "cpu".
    """
    override = os.environ.get("GPT4ALL_DEVICE", "").strip().lower()
    if override in ("cuda", "gpu", "cpu"):
        logger.info(f"Using LLM device from GPT4ALL_DEVICE: {override}")
        return override

    system = sys.platform.lower()
    machine = platform.machine().lower()

    if system == "darwin" and machine in ("arm64", "aarch64"):
        # Apple Silicon (M1/M2/M3/M4): use Metal
        device = "gpu"
        logger.info("Detected macOS on Apple Silicon; using device='gpu' (Metal).")
        return device

    if system in ("win32", "linux"):
        # Desktop: prefer CUDA for NVIDIA GPU
        device = "cuda"
        logger.info(
            "Detected Windows/Linux; using device='cuda'. "
            "Install with: pip install \"gpt4all[cuda]\" for GPU support."
        )
        return device

    device = "cpu"
    logger.info("Using default device='cpu'.")
    return device


def print_device_status() -> str:
    """
    Return the LLM device that will be used and, on Windows/Linux, a hint to verify GPU.
    Call this (or print(print_device_status())) to confirm GPU usage.
    """
    device = get_llm_device()
    msg = f"LLM device: {device!r}"
    if device == "cuda":
        msg += " (NVIDIA GPU). To confirm: run 'nvidia-smi' in another terminal while the model runs; GPU utilization should increase."
    elif device == "gpu":
        msg += " (Apple Metal)."
    else:
        msg += " (CPU only)."
    return msg
