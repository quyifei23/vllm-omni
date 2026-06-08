# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC handle extraction and reconstruction.

Uses ``torch.multiprocessing.reductions`` which is the stable PyTorch API
for CUDA IPC tensor sharing.  On the producer side, ``reduce_tensor``
extracts the IPC handle; on the consumer side, ``rebuild_cuda_tensor``
reconstructs the tensor (zero-copy — it views the producer's allocation).

IMPORTANT: ``rebuild_cuda_tensor`` returns a tensor that directly accesses
the producer's GPU memory.  For cuda_ipc mode this is the desired behavior.
For cuda_copy mode we immediately copy to a local tensor and close the IPC mapping.
"""
from __future__ import annotations

import torch

from .logging import get_logger

logger = get_logger(__name__)


def extract_ipc_args(tensor: torch.Tensor) -> tuple:
    """Extract CUDA IPC reconstruction args from a tensor.

    Calls ``torch.multiprocessing.reductions.reduce_tensor`` which returns
    ``(rebuild_func, args)`` where args contains the IPC handle bytes.

    The tensor's CUDA stream must be synchronized before calling this.
    """
    if not tensor.is_cuda:
        raise ValueError(
            f"extract_ipc_args requires a CUDA tensor, got device={tensor.device}"
        )

    from torch.multiprocessing.reductions import reduce_tensor

    _, args = reduce_tensor(tensor)
    logger.debug("extract_ipc_args: shape=%s dtype=%s args_len=%d",
                 tuple(tensor.shape), tensor.dtype, len(args))
    return args


def rebuild_from_ipc_args(
    ipc_args: tuple,
) -> torch.Tensor:
    """Reconstruct a tensor from IPC args (zero-copy view of sender's memory).

    The returned tensor shares memory with the producer allocation.
    Caller must ensure the producer tensor lives until the consumer
    is done with it (cuda_ipc mode) or explicitly copies the data
    (cuda_copy mode).
    """
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    try:
        tensor = rebuild_cuda_tensor(*ipc_args)
        logger.debug("rebuild_from_ipc_args: shape=%s dtype=%s device=%s",
                     tuple(tensor.shape), tensor.dtype, str(tensor.device))
        return tensor
    except Exception as e:
        raise RuntimeError(
            f"Failed to rebuild CUDA tensor from IPC args: {e}. "
            f"Check PyTorch version compatibility (requires PyTorch >= 2.0)."
        ) from e
