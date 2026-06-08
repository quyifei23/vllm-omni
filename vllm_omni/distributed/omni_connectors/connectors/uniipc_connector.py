# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UniIPC connector: SHM/inline for metadata + CUDA IPC for GPU tensors.

Composes ``SharedMemoryConnector`` for standard metadata serialization
and uses the ``gpu_transport`` package to keep GPU tensors on-device
across process boundaries.
"""
from __future__ import annotations

import os
import time
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)


class UniIPCConnector(OmniConnectorBase):
    """GPU tensor-aware connector using composition with SharedMemoryConnector.

    Configuration keys (in ``extra`` dict of ConnectorSpec):

    - ``stage_id`` (int): stage identifier (passed to SHM connector)
    - ``device`` (str): default device, e.g. ``"cuda:0"``
    - ``shm_threshold_bytes`` (int): inline-vs-SHM threshold (default 65536)
    - ``gpu_transport_mode`` (str): ``"cuda_ipc"`` or ``"cuda_copy"``
    - ``src_device`` (int): producer GPU index
    - ``dst_device`` (int): consumer GPU index
    - ``release_timeout_ms`` (float): registry timeout (default 10000)
    """

    # Class-level ACK pipe registry.  Populated by the orchestrator
    # before stage workers are forked so the values survive fork.
    _stage_ack_pipes: dict[int, dict[str, Any]] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)

        # SHM / inline delegate
        self._shm = SharedMemoryConnector(config)

        # GPU transport config
        self._transport_mode: str = config.get("gpu_transport_mode", "cuda_ipc")
        self._src_device: int = int(config.get("src_device", 0))
        self._dst_device: int = int(config.get("dst_device", 1))
        self._release_timeout_ms: float = float(
            config.get("release_timeout_ms", 10_000.0))
        self._transport: Any = None  # lazy-init

        # Size threshold: skip GPU transport for small tensors (0 = disabled)
        self._gpu_transport_min_bytes: int = int(
            config.get("gpu_transport_min_bytes", 65536))

        # Memory pressure threshold: fall back to inline when free ratio < threshold
        # 0.0 disables pressure checking
        self._gpu_memory_pressure_threshold: float = float(
            config.get("gpu_memory_pressure_threshold", 0.0))

        # ACK connections for GPU tensor release notifications
        self._ack_conn: Any = config.get("ack_conn", None)  # producer side: receive ACKs
        self._consumer_ack_conn: Any = config.get("consumer_ack_conn", None)  # consumer side: send ACKs

        self._metrics: dict[str, Any] = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_sent": 0,
            "gpu_tensors_recv": 0,
            "gpu_tensors_inlined": 0,
            "inline_bytes": 0,
            "pressure_fallbacks": 0,
            "put_total_ms": 0.0,
            "get_total_ms": 0.0,
        }

        # Track GPU transport tensor_ids by put_key for per-request ACK.
        # Key: put_key (e.g. "req-123_0_0"), Value: list of tensor_id strings.
        self._pending_gpu_tensors: dict[str, list[str]] = {}

        # Track GPU transport tensor_ids received via get() for per-request ACK.
        # Key: get_key, Value: list of tensor_id strings.
        self._received_gpu_tensors: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_transport(self):
        """Lazy-init the GPU transport.

        Uses importlib to load gpu_transport submodules without triggering
        the full ``vllm_omni.__init__`` import chain.
        """
        if self._transport is not None:
            return

        # "none" mode: no transport needed, router always returns "inline"
        if self._transport_mode == "none":
            return

        import importlib.util
        import sys

        gpu_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "gpu_transport",
        )

        def _load(name, fname):
            full = f"vllm_omni.distributed.gpu_transport.{name}"
            if full in sys.modules:
                return sys.modules[full]
            path = os.path.join(gpu_base, fname)
            spec = importlib.util.spec_from_file_location(full, path)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "vllm_omni.distributed.gpu_transport"
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            return mod

        _load("config", "config.py")
        # Load dependencies needed by create_transport imports
        _load("logging", "logging.py")
        _load("protocol", "protocol.py")
        _load("tensor_registry", "tensor_registry.py")
        _load("ipc_utils", "ipc_utils.py")
        _load("cuda_ipc_transport", "cuda_ipc_transport.py")
        _load("cuda_copy_transport", "cuda_copy_transport.py")

        # Ensure the gpu_transport package itself is loaded (not a stub),
        # so create_transport is importable from it.
        pkg_name = "vllm_omni.distributed.gpu_transport"
        pkg_path = os.path.join(gpu_base, "__init__.py")
        if os.path.isfile(pkg_path):
            spec = importlib.util.spec_from_file_location(pkg_name, pkg_path)
            pkg_mod = importlib.util.module_from_spec(spec)
            pkg_mod.__package__ = pkg_name
            pkg_mod.__path__ = [gpu_base]
            sys.modules[pkg_name] = pkg_mod
            spec.loader.exec_module(pkg_mod)

        from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
        from vllm_omni.distributed.gpu_transport import create_transport

        transport_config = GPUTransportConfig(
            mode=self._transport_mode,
            src_device=self._src_device,
            dst_device=self._dst_device,
            release_timeout_ms=self._release_timeout_ms,
            ack_conn=self._ack_conn,
            consumer_ack_conn=self._consumer_ack_conn,
        )
        self._transport = create_transport(transport_config)

        # Start ACK thread if transport supports it and ack_conn is wired
        if self._transport is not None and self._ack_conn is not None:
            if hasattr(self._transport, '_start_ack_thread'):
                self._transport._start_ack_thread()

    def _route_tensor(self, tensor: "torch.Tensor") -> str:
        """Decide transport strategy for a single GPU tensor.

        Returns ``"inline"`` to serialize to CPU bytes in the marker,
        or ``"ipc"`` to use the GPU transport (CUDA IPC / copy).

        Checks are evaluated in order: mode, size, memory pressure.
        """
        import torch

        nbytes = tensor.numel() * tensor.element_size()

        # Dimension 1: mode "none" always inlines
        if self._transport_mode == "none":
            self._metrics["gpu_tensors_inlined"] += 1
            self._metrics["inline_bytes"] += nbytes
            return "inline"

        # Dimension 2: size threshold
        if nbytes < self._gpu_transport_min_bytes:
            self._metrics["gpu_tensors_inlined"] += 1
            self._metrics["inline_bytes"] += nbytes
            return "inline"

        # Dimension 3: memory pressure (producer-side only for cuda_ipc)
        if (self._transport_mode == "cuda_ipc"
                and self._gpu_memory_pressure_threshold > 0.0):
            free, total = torch.cuda.mem_get_info(tensor.device)
            ratio = free / total
            if ratio < self._gpu_memory_pressure_threshold:
                self._metrics["gpu_tensors_inlined"] += 1
                self._metrics["inline_bytes"] += nbytes
                self._metrics["pressure_fallbacks"] += 1
                return "inline"

        return "ipc"

    @staticmethod
    def _has_gpu(obj: Any) -> bool:
        """Return True if *obj* contains any CUDA tensors (inline, no imports)."""
        import torch
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            return True
        if isinstance(obj, dict):
            return any(UniIPCConnector._has_gpu(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(UniIPCConnector._has_gpu(v) for v in obj)
        return False

    def _split(self, obj: Any) -> Any:
        """Replace GPU tensors with ``__gpux__`` markers using the transport."""
        if not self._has_gpu(obj):
            return obj
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            split_gpu_tensors)
        obj = split_gpu_tensors(
            obj, self._transport,
            router=self._route_tensor,
            dst_device=f"cuda:{self._dst_device}",
        )
        self._metrics["gpu_tensors_sent"] += 1
        return obj

    def _reassemble(self, obj: Any) -> Any:
        """Replace ``__gpux__`` markers with real GPU tensors."""
        if not self._has_markers(obj):
            return obj
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            reassemble_gpu_tensors)
        obj = reassemble_gpu_tensors(obj, self._transport)
        self._metrics["gpu_tensors_recv"] += 1
        return obj

    @staticmethod
    def _has_markers(obj: Any) -> bool:
        """Return True if *obj* contains any ``__gpux__`` markers."""
        if isinstance(obj, dict):
            if obj.get("__gpux__"):
                return True
            return any(UniIPCConnector._has_markers(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(UniIPCConnector._has_markers(v) for v in obj)
        return False

    @staticmethod
    def _collect_tensor_ids(obj: Any) -> list[str]:
        """Walk *obj* and collect all tensor_ids from ``__gpux__`` markers."""
        ids: list[str] = []
        if isinstance(obj, dict):
            if obj.get("__gpux__"):
                tid = obj.get("tensor_id")
                if tid:
                    ids.append(tid)
            else:
                for v in obj.values():
                    ids.extend(UniIPCConnector._collect_tensor_ids(v))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                ids.extend(UniIPCConnector._collect_tensor_ids(v))
        return ids

    # ------------------------------------------------------------------
    # OmniConnectorBase interface
    # ------------------------------------------------------------------

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Split GPU tensors, delegate metadata to SHM connector."""
        t0 = time.perf_counter()
        tensor_ids = None
        try:
            stripped = self._split(data)
            # Track GPU transport tensor_ids for per-request ACK
            tensor_ids = self._collect_tensor_ids(stripped)
            if tensor_ids:
                self._pending_gpu_tensors[put_key] = tensor_ids
            success, size, metadata = self._shm.put(
                from_stage, to_stage, put_key, stripped)
            if not success:
                # SHM write failed — release GPU tensors already registered
                # by _split(), otherwise they leak in the transport registry.
                self._release_registered_tensors(put_key, tensor_ids)
                return False, 0, None
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            self._metrics["put_total_ms"] += (time.perf_counter() - t0) * 1000.0
            return True, size, metadata
        except Exception:
            logger.exception("UniIPC put failed for key=%s", put_key)
            self._release_registered_tensors(put_key, tensor_ids)
            return False, 0, None

    def _release_registered_tensors(
        self, put_key: str, tensor_ids: list[str] | None,
    ) -> None:
        """Release GPU transport registrations from a failed put."""
        if tensor_ids:
            self._pending_gpu_tensors.pop(put_key, None)
        if tensor_ids and self._transport is not None:
            for tid in tensor_ids:
                try:
                    self._transport.release(tid)
                except Exception:
                    pass

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Retrieve via SHM connector, then reassemble GPU tensors.

        ACKs each GPU tensor immediately after reassembly so the
        producer can release its TensorRegistry entry without waiting
        for request completion.
        """
        t0 = time.perf_counter()
        try:
            result = self._shm.get(from_stage, to_stage, get_key, metadata)
            if result is None:
                return None
            obj, size = result
            # Collect GPU tensor IDs before reassembly replaces __gpux__
            # markers, then ACK immediately.
            tensor_ids = None
            if self._has_markers(obj) and self._transport_mode == "cuda_ipc":
                tensor_ids = self._collect_tensor_ids(obj)
            obj = self._reassemble(obj)
            if tensor_ids:
                for tid in tensor_ids:
                    self.notify_gpu_tensor_consumed(tid)
            self._metrics["gets"] += 1
            self._metrics["get_total_ms"] += (time.perf_counter() - t0) * 1000.0
            return obj, size
        except Exception:
            logger.exception("UniIPC get failed for key=%s", get_key)
            return None

    def release_gpu_tensors(self, request_id: str) -> None:
        """Release GPU transport tensors for *request_id* via Pipe ACK."""
        prefix = f"{request_id}_"
        # Producer side: tensors sent via put()
        put_keys = [k for k in list(self._pending_gpu_tensors) if k.startswith(prefix)]
        recv_keys = [k for k in list(self._received_gpu_tensors) if k.startswith(prefix)]
        total_tids = sum(
            len(self._pending_gpu_tensors.get(k, [])) for k in put_keys
        ) + sum(
            len(self._received_gpu_tensors.get(k, [])) for k in recv_keys
        )
        if total_tids:
            logger.debug(
                "release_gpu_tensors: req=%s put_keys=%d recv_keys=%d total_tids=%d",
                request_id, len(put_keys), len(recv_keys), total_tids,
            )
        for key in put_keys:
            for tid in self._pending_gpu_tensors.pop(key, []):
                self.notify_gpu_tensor_consumed(tid)
        for key in recv_keys:
            for tid in self._received_gpu_tensors.pop(key, []):
                self.notify_gpu_tensor_consumed(tid)

    def notify_gpu_tensor_consumed(self, tensor_id: str) -> None:
        """Notify producer that a GPU tensor (cuda_ipc mode) is no longer needed."""
        if self._transport is not None and hasattr(self._transport, 'notify_consumed'):
            self._transport.notify_consumed(tensor_id)

    def cleanup(self, request_id: str) -> None:
        """Clean SHM segments and release transport-held tensors."""
        self._shm.cleanup(request_id)
        self.release_gpu_tensors(request_id)

    def close(self) -> None:
        """Release SHM connector and GPU transport."""
        # Release all tracked GPU tensors before closing transport
        for key in list(self._pending_gpu_tensors):
            for tid in self._pending_gpu_tensors.pop(key, []):
                self.notify_gpu_tensor_consumed(tid)
        for key in list(self._received_gpu_tensors):
            for tid in self._received_gpu_tensors.pop(key, []):
                self.notify_gpu_tensor_consumed(tid)
        self._shm.close()
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def set_ack_conns(self, ack_conn: Any, consumer_ack_conn: Any) -> None:
        """Wire ACK pipe connections after construction.

        Called after the connector is created, so that the
        ``multiprocessing.Connection`` objects never enter vLLM's
        config hash computation.
        """
        self._ack_conn = ack_conn
        self._consumer_ack_conn = consumer_ack_conn

    @property
    def dst_device(self) -> str:
        """Configured destination GPU device string (e.g. ``cuda:7``)."""
        return f"cuda:{self._dst_device}"

    def health(self) -> dict[str, Any]:
        result = {
            "status": "healthy",
            "transport_mode": self._transport_mode,
            "src_device": self._src_device,
            "dst_device": self._dst_device,
            "gpu_transport_min_bytes": self._gpu_transport_min_bytes,
            "gpu_memory_pressure_threshold": self._gpu_memory_pressure_threshold,
            **self._metrics,
        }
        # Add real-time registry size if transport is active
        if self._transport is not None and hasattr(self._transport, 'registry_size'):
            result["current_registry_size"] = self._transport.registry_size
        else:
            result["current_registry_size"] = 0
        # Add current GPU free memory ratio
        try:
            import torch
            free, total = torch.cuda.mem_get_info(self._src_device)
            result["gpu_free_memory_ratio"] = round(free / total, 4)
        except Exception:
            result["gpu_free_memory_ratio"] = 0.0
        return result
