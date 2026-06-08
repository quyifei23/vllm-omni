# SPDX-License-Identifier: Apache-2.0
"""Common utilities for Motivation 2 experiments."""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import torch

try:
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
except ImportError:
    OmniSerializer = None

try:
    from vllm_omni.entrypoints.stage_utils import shm_write_bytes as _shm_write_bytes
    from vllm_omni.entrypoints.stage_utils import shm_read_bytes as _shm_read_bytes
except ImportError:
    _shm_write_bytes = None
    _shm_read_bytes = None

# Path labels used throughout experiments
PATH_LABELS = {
    "inline": "Inline / Queue Baseline",
    "serialized_shm": "Serialized SHM (Current Default)",
    "raw_shm": "Raw CPU SHM (No Serialization)",
    "cuda_ipc": "CUDA IPC Baseline",
    "cuda_ipc_d2d": "CUDA IPC D2D Copy",
    "mooncake": "Mooncake Fast Path",
    "ucx": "UCX / RDMA",
}

PAYLOAD_CATEGORIES = [
    "metadata",
    "cpu_tensor",
    "gpu_tensor",
    "mixed_cpu",
    "mixed_gpu",
]


def compute_stats(latencies_us: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of latencies (in microseconds)."""
    if not latencies_us:
        return {}
    sorted_lats = sorted(latencies_us)
    n = len(sorted_lats)
    return {
        "count": n,
        "mean_us": statistics.mean(latencies_us),
        "std_us": statistics.stdev(latencies_us) if n > 1 else 0.0,
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
        "p50_us": sorted_lats[int(n * 0.50)],
        "p90_us": sorted_lats[int(n * 0.90)],
        "p99_us": sorted_lats[int(n * 0.99)],
    }


def format_bytes(size_bytes: int) -> str:
    """Human-readable byte size."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def cpu_affinity(core: int) -> None:
    """Set CPU affinity to a specific core."""
    try:
        os = __import__("os")
        os.sched_setaffinity(0, {core})
    except Exception:
        pass


def sync_cuda() -> None:
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def serialize_payload(obj: Any) -> bytes:
    """Serialize using the OmniSerializer."""
    if OmniSerializer is None:
        raise ImportError("vllm_omni not available for serialization")
    return OmniSerializer.serialize(obj)


def deserialize_payload(data: bytes) -> Any:
    """Deserialize using the OmniSerializer."""
    if OmniSerializer is None:
        raise ImportError("vllm_omni not available for deserialization")
    return OmniSerializer.deserialize(data)


def shm_write(data: bytes, name: str | None = None) -> dict[str, Any]:
    """Write bytes to POSIX shared memory."""
    if _shm_write_bytes is None:
        raise ImportError("vllm_omni SHM utilities not available")
    return _shm_write_bytes(data, name=name)


def shm_read(meta: dict[str, Any]) -> bytes:
    """Read bytes from POSIX shared memory."""
    if _shm_read_bytes is None:
        raise ImportError("vllm_omni SHM utilities not available")
    return _shm_read_bytes(meta)
