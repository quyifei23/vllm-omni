# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload split/reassemble for GPU tensor transport.

Walks a nested dict/list structure, replaces GPU tensors with lightweight
``__gpux__`` markers (safe for Pipe/msgpack serialization), and restores
them on the receiver side.
"""
from __future__ import annotations

import io
import pickle
import uuid
from typing import Any

import torch

from .protocol import TensorMetadata, TransportHandle
from .logging import get_logger

logger = get_logger(__name__)

_GPUX_MARKER = "__gpux__"


def has_gpu_tensors(obj: Any) -> bool:
    """Return True if *obj* contains any CUDA tensors."""
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        return True
    if isinstance(obj, dict):
        return any(has_gpu_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_gpu_tensors(v) for v in obj)
    return False


def _make_inline_marker(tensor: torch.Tensor, dst_device: str) -> dict:
    """Serialize a GPU tensor to CPU bytes and wrap in a ``__gpux__`` marker."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    buf = io.BytesIO()
    torch.save(cpu_tensor, buf)
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    return {
        _GPUX_MARKER: True,
        "tensor_id": uuid.uuid4().hex[:12],
        "meta": {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "nbytes": nbytes,
            "dst_device": dst_device,
        },
        "inline_data": buf.getvalue(),
    }


def _resolve_local_device(dst_device: str) -> torch.device:
    """Resolve *dst_device* to a valid local CUDA device.

    In multi-stage deployments the producer sets ``dst_device`` to the
    physical GPU index (e.g. ``cuda:7``), but vLLM may remap that GPU
    to a different index in the consumer process.  Falls back to the
    current CUDA device when the requested device is not available.
    """
    try:
        dev = torch.device(dst_device)
        # Trigger device context to validate the ordinal is reachable.
        with torch.cuda.device(dev):
            pass
        return dev
    except (RuntimeError, torch.AcceleratorError):
        return torch.device(torch.cuda.current_device())


def _recover_inline_tensor(marker: dict) -> torch.Tensor:
    """Recover a GPU tensor from an inline marker."""
    buf = io.BytesIO(marker["inline_data"])
    tensor = torch.load(buf, weights_only=True)
    dst_device = marker["meta"].get("dst_device", "cuda:0")
    return tensor.to(_resolve_local_device(dst_device))


def split_gpu_tensors(
    obj: Any,
    transport: Any,
    router: Any = None,
    dst_device: str | None = None,
) -> Any:
    """Walk *obj* and replace every CUDA tensor with a ``__gpux__`` marker.

    If *router* is provided, it is called for each tensor and must return
    ``"inline"`` or ``"ipc"``.  ``"inline"`` serializes the tensor to CPU
    bytes via ``_make_inline_marker``; ``"ipc"`` uses the GPU transport.

    If *router* is ``None``, all tensors go through the GPU transport
    (original behaviour).
    """
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        if router is not None:
            decision = router(obj)
            if decision == "inline":
                return _make_inline_marker(
                    obj, dst_device=dst_device or str(obj.device)
                )
        # Original IPC path
        handle = transport.send(obj, dst_rank=transport._config.dst_device)
        handle.metadata.ipc_args = transport._ipc_args_store[handle.tensor_id]
        marker = {
            _GPUX_MARKER: True,
            "tensor_id": handle.tensor_id,
            "meta": handle.metadata.to_dict(),
            "ipc_args": pickle.dumps(handle.metadata.ipc_args),
        }
        logger.debug(
            "split: replaced tensor id=%s shape=%s",
            handle.tensor_id,
            handle.metadata.shape,
        )
        return marker

    if isinstance(obj, dict):
        return {k: split_gpu_tensors(v, transport, router=router, dst_device=dst_device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(split_gpu_tensors(v, transport, router=router, dst_device=dst_device) for v in obj)

    return obj


def reassemble_gpu_tensors(obj: Any, transport: Any) -> Any:
    """Walk *obj* and replace every ``__gpux__`` marker with a real GPU tensor.

    Handles both IPC markers (original path) and inline markers
    (``inline_data`` key present).
    """
    if isinstance(obj, dict) and obj.get(_GPUX_MARKER):
        # Inline path: recover from CPU bytes
        if "inline_data" in obj:
            return _recover_inline_tensor(obj)

        # IPC path (original)
        meta = TensorMetadata.from_dict(obj["meta"])
        meta.ipc_args = (
            pickle.loads(obj["ipc_args"])
            if isinstance(obj["ipc_args"], bytes)
            else obj["ipc_args"]
        )
        handle = TransportHandle(tensor_id=obj["tensor_id"], metadata=meta)
        return transport.recv(
            handle,
            src_rank=transport._config.src_device,
            dst_device=_resolve_local_device(meta.dst_device),
        )

    if isinstance(obj, dict):
        return {k: reassemble_gpu_tensors(v, transport) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(reassemble_gpu_tensors(v, transport) for v in obj)

    return obj
