# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .protocol import GPUTensorTransport, TensorMetadata, TransportHandle
from .config import GPUTransportConfig, TransportMode
from .logging import get_logger
from .cuda_ipc_transport import CudaIpcTransport
from .cuda_copy_transport import CudaCopyTransport
from .split import split_gpu_tensors, reassemble_gpu_tensors, has_gpu_tensors


def create_transport(config: GPUTransportConfig):
    """Factory: returns the correct transport based on config.mode."""
    if config.mode == "cuda_ipc":
        return CudaIpcTransport(config)
    elif config.mode == "cuda_copy":
        return CudaCopyTransport(config)
    elif config.mode == "none":
        return None
    raise ValueError(f"Unknown transport mode: {config.mode}")


__all__ = [
    "GPUTensorTransport", "TensorMetadata", "TransportHandle",
    "GPUTransportConfig", "TransportMode", "get_logger",
    "CudaIpcTransport", "CudaCopyTransport", "create_transport",
    "split_gpu_tensors", "reassemble_gpu_tensors", "has_gpu_tensors",
]
