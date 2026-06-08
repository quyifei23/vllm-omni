# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strategy 2: Eager P2P copy via CUDA IPC + cudaMemcpy.

Consumer opens producer's allocation via IPC, immediately copies data
to a local tensor on the destination GPU using a dedicated CUDA stream,
then sends copy_done ACK so the producer can release the original tensor.
Consumer uses its local copy — no further dependency on producer allocation.
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


class CudaCopyTransport:
    """Eager P2P copy CUDA IPC transport."""

    def __init__(self, config: GPUTransportConfig):
        self._config = config
        self._registry = TensorRegistry(timeout_ms=config.release_timeout_ms)
        self._ipc_args_store: dict[str, tuple] = {}
        self._copy_streams: dict[str, torch.cuda.Stream] = {}
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
        if not hasattr(CudaCopyTransport, "_libcudart"):
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
                CudaCopyTransport._libcudart = None
                return
            lib = ctypes.CDLL(lib_path)
            lib.cudaSetDevice.argtypes = [ctypes.c_int]
            lib.cudaSetDevice.restype = ctypes.c_int
            lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
            lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
            CudaCopyTransport._libcudart = lib

        lib = CudaCopyTransport._libcudart
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

    @staticmethod
    def _resolve_local_ordinals(
        src_physical: int, dst_dev: torch.device,
    ) -> tuple[int | None, int | None]:
        """Map physical GPU indices to local device ordinals.

        When CUDA_VISIBLE_DEVICES remaps GPUs in a consumer process the
        physical indices stored in config no longer match local ordinals.
        Each element is ``None`` when the corresponding device is not
        locally visible.
        """
        def _resolve_one(physical_idx: int) -> int | None:
            try:
                dev = torch.device(f"cuda:{physical_idx}")
                with torch.cuda.device(dev):
                    pass
                return dev.index
            except (RuntimeError, torch.AcceleratorError):
                return None

        src_local = _resolve_one(src_physical)
        # dst_dev may already carry a remapped ordinal; validate it
        try:
            with torch.cuda.device(dst_dev):
                pass
            dst_local = dst_dev.index
        except (RuntimeError, torch.AcceleratorError):
            dst_local = None
        return src_local, dst_local

    def _start_ack_thread(self) -> None:
        """Start the ACK thread if ack_conn is set. Idempotent."""
        if self._ack_conn is not None and not self._ack_running:
            self._ack_running = True
            self._ack_thread = threading.Thread(target=self._ack_loop, daemon=True)
            self._ack_thread.start()

    def _get_copy_stream(self, device: str) -> torch.cuda.Stream:
        if device not in self._copy_streams:
            self._copy_streams[device] = torch.cuda.Stream(
                device=torch.device(device))
        return self._copy_streams[device]

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
            mode="cuda_copy",
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
        if meta.mode != "cuda_copy":
            raise ValueError(f"Expected mode=cuda_copy, got {meta.mode}")

        ipc_args = meta.ipc_args
        if ipc_args is None:
            raise ValueError("TransportHandle has no IPC args")

        dst_dev = torch.device(dst_device) if isinstance(dst_device, str) else dst_device

        # Resolve both source and destination to local device ordinals.
        # In deployments with CUDA_VISIBLE_DEVICES remapping the physical
        # GPU indices stored in config do not match local ordinals.
        src_local, dst_local = self._resolve_local_ordinals(self._config.src_device, dst_dev)

        # Check P2P capability (skip for same-device, device can always access itself)
        if src_local is not None and src_local != dst_local:
            try:
                has_peer = torch.cuda.can_device_access_peer(src_local, dst_local)
            except (RuntimeError, AssertionError):
                has_peer = False
            if not has_peer:
                raise RuntimeError(
                    f"P2P access not available from device {src_local} "
                    f"to {dst_local}. Enable peer access or use cuda_ipc mode instead."
                )

        t0 = time.perf_counter()
        # 1. Open IPC allocation (zero-copy view of producer memory)
        with torch.cuda.device(dst_dev):
            src_tensor = rebuild_from_ipc_args(ipc_args)
        t1 = time.perf_counter()

        # 2. Allocate local tensor on dst_device
        torch_dtype = getattr(torch, meta.dtype.replace("torch.", ""))
        local_tensor = torch.empty(meta.shape, dtype=torch_dtype, device=dst_dev)
        t2 = time.perf_counter()

        # 3. Async P2P copy on dedicated stream, then synchronize
        copy_stream = self._get_copy_stream(str(dst_dev))
        with torch.cuda.stream(copy_stream):
            local_tensor.copy_(src_tensor, non_blocking=True)
            copy_event = torch.cuda.Event()
            copy_event.record(copy_stream)
        copy_event.synchronize()
        t3 = time.perf_counter()

        # 4. Send copy_done ACK so producer can release the original tensor
        if self._consumer_ack_conn is not None:
            try:
                from .control_channel import ConsumerControl
                ctrl = ConsumerControl(self._consumer_ack_conn)
                ctrl.send_ack(meta.tensor_id, ack_type="copy_done")
                logger.debug("recv: sent copy_done ACK for id=%s", meta.tensor_id)
            except Exception:
                logger.warning("recv: failed to send copy_done ACK for id=%s",
                               meta.tensor_id, exc_info=True)

        logger.debug(
            "recv: id=%s shape=%s open_ms=%.3f alloc_ms=%.3f copy_ms=%.3f",
            meta.tensor_id, meta.shape,
            (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
        )
        return local_tensor

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

    def release(self, tensor_id: str) -> None:
        self._ipc_args_store.pop(tensor_id, None)
        self._registry.release(tensor_id)

    def close(self) -> None:
        self.shutdown_ack_thread()
        if self._ack_thread is not None and self._ack_thread.is_alive():
            self._ack_thread.join(timeout=2.0)
        self._ipc_args_store.clear()
        self._registry.clear()
        for s in self._copy_streams.values():
            try:
                s.synchronize()
            except Exception:
                pass
        self._copy_streams.clear()

    def cleanup_timeouts(self) -> list[str]:
        return self._registry.cleanup_timeouts()

    @property
    def registry_size(self) -> int:
        return self._registry.size
