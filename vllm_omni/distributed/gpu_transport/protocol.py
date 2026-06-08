# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class TensorMetadata:
    """Lightweight metadata about a GPU tensor being transported."""
    tensor_id: str
    mode: str
    src_device: str
    dst_device: str
    shape: list[int]
    dtype: str
    nbytes: int
    contiguous: bool = True
    # CUDA IPC reconstruction args from torch.multiprocessing.reductions
    ipc_args: tuple | None = None
    # Timestamps in seconds
    send_start_ts: float = 0.0
    ipc_meta_ready_ts: float = 0.0

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        tensor_id: str | None = None,
        mode: str = "cuda_ipc",
        src_device: str = "cuda:0",
        dst_device: str = "cuda:1",
    ) -> "TensorMetadata":
        if not tensor.is_cuda:
            raise ValueError(f"Only CUDA tensors supported, got device={tensor.device}")
        t = tensor if tensor.is_contiguous() else tensor.contiguous()
        return cls(
            tensor_id=tensor_id or uuid.uuid4().hex[:12],
            mode=mode,
            src_device=src_device,
            dst_device=dst_device,
            shape=list(t.shape),
            dtype=str(t.dtype),
            nbytes=t.numel() * t.element_size(),
            contiguous=tensor.is_contiguous(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for pipe transport (no pickle of tensor data)."""
        return {
            "tensor_id": self.tensor_id,
            "mode": self.mode,
            "src_device": self.src_device,
            "dst_device": self.dst_device,
            "shape": self.shape,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "contiguous": self.contiguous,
            "send_start_ts": self.send_start_ts,
            "ipc_meta_ready_ts": self.ipc_meta_ready_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TensorMetadata":
        return cls(
            tensor_id=d["tensor_id"],
            mode=d["mode"],
            src_device=d["src_device"],
            dst_device=d["dst_device"],
            shape=d["shape"],
            dtype=d["dtype"],
            nbytes=d["nbytes"],
            contiguous=d.get("contiguous", True),
            send_start_ts=d.get("send_start_ts", 0.0),
            ipc_meta_ready_ts=d.get("ipc_meta_ready_ts", 0.0),
        )


@dataclass
class TransportHandle:
    """Opaque handle from send(), consumed by recv()."""
    tensor_id: str
    metadata: TensorMetadata


@runtime_checkable
class GPUTensorTransport(Protocol):
    """Protocol for GPU tensor cross-process transport."""

    def send(
        self,
        tensor: torch.Tensor,
        *,
        dst_rank: int,
        tensor_id: str | None = None,
    ) -> TransportHandle:
        ...

    def recv(
        self,
        handle: TransportHandle,
        *,
        src_rank: int,
        dst_device: torch.device | str,
    ) -> torch.Tensor:
        ...

    def release(self, tensor_id: str) -> None:
        ...

    def close(self) -> None:
        ...
