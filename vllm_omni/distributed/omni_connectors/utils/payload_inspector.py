# SPDX-License-Identifier: Apache-2.0
"""Payload inspector for OmniConnector profiling.

Recursively inspects Python objects to derive payload statistics without
triggering GPU→CPU copies or modifying tensor content.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch


def inspect_payload(obj: Any, max_depth: int = 10) -> dict:
    """Recursively inspect a payload object for size, type, and location stats.

    Only reads tensor metadata (device, shape, dtype, numel, element_size).
    Never calls .cpu(), .numpy(), .clone(), or .contiguous().

    Returns:
        dict with keys: payload_type_summary, payload_size_bytes, tensor_count,
        cpu_tensor_bytes, gpu_tensor_bytes, bytes_size, numpy_bytes,
        contains_gpu_tensor, memory_location, max_tensor_size_bytes,
        object_depth
    """
    state = {
        "tensor_count": 0,
        "cpu_tensor_bytes": 0,
        "gpu_tensor_bytes": 0,
        "bytes_size": 0,
        "numpy_bytes": 0,
        "max_tensor_size_bytes": 0,
        "max_depth_reached": 0,
        "types_seen": set(),
        "contains_gpu_tensor": False,
    }
    _inspect_recursive(obj, depth=0, max_depth=max_depth, state=state)

    payload_size_bytes = (
        state["cpu_tensor_bytes"]
        + state["gpu_tensor_bytes"]
        + state["bytes_size"]
        + state["numpy_bytes"]
    )

    # Determine memory location
    if state["gpu_tensor_bytes"] > 0 and state["cpu_tensor_bytes"] > 0:
        memory_location = "mixed"
    elif state["gpu_tensor_bytes"] > 0:
        memory_location = "gpu"
    elif state["cpu_tensor_bytes"] > 0:
        memory_location = "cpu"
    else:
        memory_location = "cpu"  # bytes/numpy default to CPU

    # Classify payload type
    payload_type_summary = _classify_payload(state, obj)

    return {
        "payload_type_summary": payload_type_summary,
        "payload_size_bytes": payload_size_bytes,
        "tensor_count": state["tensor_count"],
        "cpu_tensor_bytes": state["cpu_tensor_bytes"],
        "gpu_tensor_bytes": state["gpu_tensor_bytes"],
        "bytes_size": state["bytes_size"],
        "numpy_bytes": state["numpy_bytes"],
        "contains_gpu_tensor": state["contains_gpu_tensor"],
        "memory_location": memory_location,
        "max_tensor_size_bytes": state["max_tensor_size_bytes"],
        "object_depth": state["max_depth_reached"],
    }


def _inspect_recursive(obj: Any, depth: int, max_depth: int, state: dict) -> None:
    if depth > max_depth:
        return
    if depth > state["max_depth_reached"]:
        state["max_depth_reached"] = depth

    if isinstance(obj, torch.Tensor):
        state["tensor_count"] += 1
        nbytes = obj.numel() * obj.element_size()
        state["max_tensor_size_bytes"] = max(state["max_tensor_size_bytes"], nbytes)
        if obj.is_cuda:
            state["gpu_tensor_bytes"] += nbytes
            state["contains_gpu_tensor"] = True
        else:
            state["cpu_tensor_bytes"] += nbytes
        state["types_seen"].add("torch.Tensor")
        return

    if isinstance(obj, np.ndarray):
        state["numpy_bytes"] += obj.nbytes
        state["types_seen"].add("numpy.ndarray")
        if obj.nbytes > state["max_tensor_size_bytes"]:
            state["max_tensor_size_bytes"] = obj.nbytes
        return

    if isinstance(obj, (bytes, bytearray)):
        state["bytes_size"] += len(obj)
        state["types_seen"].add("bytes")
        return

    if isinstance(obj, memoryview):
        state["bytes_size"] += obj.nbytes
        state["types_seen"].add("memoryview")
        return

    if isinstance(obj, (int, float, bool, str, type(None))):
        state["types_seen"].add(type(obj).__name__)
        return

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            for field in dataclasses.fields(obj):
                _inspect_recursive(getattr(obj, field.name), depth + 1, max_depth, state)
        except Exception:
            state["types_seen"].add("dataclass(inspect_error)")
        return

    if isinstance(obj, dict):
        state["types_seen"].add("dict")
        for k, v in obj.items():
            _inspect_recursive(k, depth + 1, max_depth, state)
            _inspect_recursive(v, depth + 1, max_depth, state)
        return

    if isinstance(obj, (list, tuple, set)):
        state["types_seen"].add(type(obj).__name__)
        for item in obj:
            _inspect_recursive(item, depth + 1, max_depth, state)
        return

    state["types_seen"].add(f"unknown:{type(obj).__name__}")


def _classify_payload(state: dict, obj: Any) -> str:
    """Best-effort semantic classification of payload."""
    types = state["types_seen"]

    # Check key/field names for hints
    key_hints = _collect_keys(obj, max_depth=2)

    if state["gpu_tensor_bytes"] > 0 and state["cpu_tensor_bytes"] > 0:
        return "mixed"
    if state["gpu_tensor_bytes"] > 0:
        return "tensor"
    if state["cpu_tensor_bytes"] > 0:
        # Check for KV cache hints
        if any("kv" in k.lower() or "cache" in k.lower() for k in key_hints):
            return "kv_cache"
        if any("embed" in k.lower() for k in key_hints):
            return "embedding"
        if any("audio" in k.lower() or "code" in k.lower() for k in key_hints):
            return "audio_chunk"
        if any("diffusion" in k.lower() or "latent" in k.lower() for k in key_hints):
            return "diffusion_tensor"
        if any("token" in k.lower() or "logit" in k.lower() for k in key_hints):
            return "tensor"
        return "tensor"

    # Purely bytes/scalar
    if state["bytes_size"] > 0:
        if any("meta" in k.lower() or "control" in k.lower() for k in key_hints):
            return "metadata"
        return "metadata"

    # Small scalar dicts
    total = state["cpu_tensor_bytes"] + state["gpu_tensor_bytes"] + state["bytes_size"] + state["numpy_bytes"]
    if "dict" in types and total == 0:
        return "control"

    return "unknown"


def _collect_keys(obj: Any, max_depth: int = 2, _depth: int = 0) -> set[str]:
    """Collect string keys from dict-like objects up to a given depth."""
    keys: set[str] = set()
    if _depth > max_depth:
        return keys
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                keys.add(k.lower())
            keys |= _collect_keys(v, max_depth, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys |= _collect_keys(item, max_depth, _depth + 1)
    return keys
