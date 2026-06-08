# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strategy 1: Zero-copy GPU tensor transport via CUDA IPC.

Producer exports IPC handle via torch.multiprocessing.reductions.
Consumer reconstructs tensor that directly views producer's allocation.
Producer retains the tensor until consumer sends release ACK.
"""
from __future__ import annotations

import threading
import time
import uuid
from multiprocessing.connection import Connection

import torch

from .protocol import GPUTensorTransport, TensorMetadata, TransportHandle
from .config import GPUTransportConfig
from .tensor_registry import TensorRegistry
from .ipc_utils import extract_ipc_args, rebuild_from_ipc_args
from .logging import get_logger

logger = get_logger(__name__)


class CudaIpcTransport:
    """Zero-copy CUDA IPC transport."""

    def __init__(self, config: GPUTransportConfig):
        self._config = config
        self._registry = TensorRegistry(timeout_ms=config.release_timeout_ms)
        self._ipc_args_store: dict[str, tuple] = {}
        if config.enable_peer_access:
            self._ensure_peer_access(config.src_device, config.dst_device)

        # ACK thread support (optional -- only when ack_conn is provided)
        self._ack_conn: Connection | None = getattr(config, 'ack_conn', None)
        self._consumer_ack_conn: Connection | None = getattr(config, 'consumer_ack_conn', None)
        self._ack_thread: threading.Thread | None = None
        self._ack_running = False
        self._start_ack_thread()

    @staticmethod
    def _ensure_peer_access(src: int, dst: int) -> None:
        """Enable P2P access between *src* and *dst* via CUDA Runtime API.

        ``torch.cuda.device(i)`` is a context manager without an
        ``enable_peer_access`` method, so we call the CUDA Runtime
        directly via ctypes.
        """
        import ctypes
        import ctypes.util
        import os

        # Locate libcudart once and cache it.
        if not hasattr(CudaIpcTransport, "_libcudart"):
            lib_path = ctypes.util.find_library("cudart")
            if not lib_path:
                for cand in (
                    "/usr/local/cuda/lib64/libcudart.so",
                    "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
                ):
                    if os.path.isfile(cand):
                        lib_path = cand
                        break
            if not lib_path:
                logger.warning("Cannot find libcudart; P2P access will not be enabled")
                CudaIpcTransport._libcudart = None
                return
            lib = ctypes.CDLL(lib_path)
            lib.cudaSetDevice.argtypes = [ctypes.c_int]
            lib.cudaSetDevice.restype = ctypes.c_int
            lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
            lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
            CudaIpcTransport._libcudart = lib

        lib = CudaIpcTransport._libcudart
        if lib is None:
            return

        for i in (src, dst):
            for j in (src, dst):
                if i == j:
                    continue
                try:
                    if torch.cuda.can_device_access_peer(i, j):
                        err = lib.cudaSetDevice(i)
                        if err != 0:
                            logger.warning(
                                "cudaSetDevice(%d) failed: %d", i, err)
                            continue
                        err = lib.cudaDeviceEnablePeerAccess(j, 0)
                        if err != 0:
                            logger.warning(
                                "cudaDeviceEnablePeerAccess(%d->%d): cudaError=%d",
                                i, j, err,
                            )
                except Exception as e:
                    logger.warning(
                        "Failed to enable P2P access device %d -> %d: %s",
                        i, j, e,
                    )

    def _start_ack_thread(self) -> None:
        """Start the ACK thread if ack_conn is set. Idempotent."""
        if self._ack_conn is not None and not self._ack_running:
            self._ack_running = True
            self._ack_thread = threading.Thread(target=self._ack_loop, daemon=True)
            self._ack_thread.start()

    def send(
        self,
        tensor: torch.Tensor,
        *,
        dst_rank: int,
        tensor_id: str | None = None,
    ) -> TransportHandle:
        if not tensor.is_cuda:
            raise ValueError(f"Only CUDA tensors supported, got device={tensor.device}")
        _was_contiguous = tensor.is_contiguous()
        if not _was_contiguous:
            logger.warning("send: non-contiguous tensor %s, calling .contiguous()", tensor_id)
            tensor = tensor.contiguous()

        tid = tensor_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        torch.cuda.current_stream(tensor.device).synchronize()
        t1 = time.perf_counter()

        ipc_args = extract_ipc_args(tensor)
        t2 = time.perf_counter()

        metadata = TensorMetadata.from_tensor(
            tensor,
            tensor_id=tid,
            mode="cuda_ipc",
            src_device=str(tensor.device),
            dst_device=f"cuda:{self._config.dst_device}",
        )
        metadata.contiguous = _was_contiguous
        metadata.send_start_ts = t0
        metadata.ipc_meta_ready_ts = t2
        metadata.ipc_args = ipc_args

        self._registry.register(tid, tensor)
        self._ipc_args_store[tid] = ipc_args

        logger.debug(
            "send: id=%s shape=%s dtype=%s nbytes=%d sync_ms=%.3f ipc_ms=%.3f",
            tid, metadata.shape, metadata.dtype, metadata.nbytes,
            (t1 - t0) * 1000, (t2 - t1) * 1000,
        )
        return TransportHandle(tensor_id=tid, metadata=metadata)

    def recv(
        self,
        handle: TransportHandle,
        *,
        src_rank: int,
        dst_device: torch.device | str,
    ) -> torch.Tensor:
        meta = handle.metadata
        if meta.mode != "cuda_ipc":
            raise ValueError(f"Expected mode=cuda_ipc, got {meta.mode}")

        ipc_args = meta.ipc_args
        if ipc_args is None:
            raise ValueError("TransportHandle has no IPC args — was metadata set by caller?")

        t0 = time.perf_counter()
        dst_dev = torch.device(dst_device) if isinstance(dst_device, str) else dst_device
        with torch.cuda.device(dst_dev):
            tensor = rebuild_from_ipc_args(ipc_args)
        t1 = time.perf_counter()

        logger.debug(
            "recv: id=%s shape=%s dtype=%s rebuild_ms=%.3f",
            meta.tensor_id, meta.shape, meta.dtype, (t1 - t0) * 1000,
        )
        return tensor

    def _ack_loop(self) -> None:
        """Background daemon thread: poll ACK pipe and auto-release tensors."""
        from .control_channel import ProducerControl
        ctrl = ProducerControl(self._ack_conn)
        while self._ack_running:
            try:
                msg = ctrl.recv_ack(timeout_ms=500.0)
            except Exception:
                logger.exception("ack_thread: unexpected error in recv_ack")
                continue
            if msg is None:
                continue
            if msg.get("type") == "shutdown":
                break
            tensor_id = msg.get("tensor_id", "")
            if tensor_id:
                logger.debug("ack_thread: releasing id=%s", tensor_id)
                self.release(tensor_id)

    def shutdown_ack_thread(self) -> None:
        """Signal the ACK thread to stop (does not join)."""
        self._ack_running = False

    def notify_consumed(self, tensor_id: str) -> None:
        """Consumer calls this when done using a zero-copy tensor.

        Sends a ``release`` ACK to the producer so it can free the tensor
        from its TensorRegistry.  Safe to call multiple times (idempotent
        on the producer side).
        """
        if self._consumer_ack_conn is None:
            return
        try:
            from .control_channel import ConsumerControl
            ctrl = ConsumerControl(self._consumer_ack_conn)
            ctrl.send_ack(tensor_id, ack_type="release")
            logger.debug("notify_consumed: sent release ACK for id=%s", tensor_id)
        except Exception:
            logger.warning("notify_consumed: failed to send ACK for id=%s",
                           tensor_id, exc_info=True)

    def release(self, tensor_id: str) -> None:
        self._ipc_args_store.pop(tensor_id, None)
        # Note: self._registry.release() is thread-safe (uses threading.Lock inside TensorRegistry).
        # The _ipc_args_store.pop() above is safe because dict.pop() is atomic at the Python
        # interpreter level (single opcode).  If concurrent access becomes an issue, extract
        # _ipc_args_store into TensorRegistry.
        self._registry.release(tensor_id)

    def close(self) -> None:
        self.shutdown_ack_thread()
        if self._ack_thread is not None and self._ack_thread.is_alive():
            self._ack_thread.join(timeout=2.0)
        self._ipc_args_store.clear()
        self._registry.clear()

    def cleanup_timeouts(self) -> list[str]:
        return self._registry.cleanup_timeouts()

    @property
    def registry_size(self) -> int:
        return self._registry.size
