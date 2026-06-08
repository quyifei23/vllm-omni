# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for GPU tensor transport system.

All cross-process tests use ``mp.set_start_method("spawn", force=True)``
because CUDA does not support fork.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import os
import sys
import types

import pytest
import torch

# ---------------------------------------------------------------------------
# Import hack: set up the vllm_omni package hierarchy in sys.modules WITHOUT
# triggering vllm_omni/__init__.py (which imports vllm, which may fail when
# its dependencies are mismatched, e.g. an incompatible transformers version).
#
# This is needed both in the pytest process AND in spawned child processes.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# tests/distributed/gpu_transport -> repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
_VLLM_OMNI_BASE = os.path.join(_REPO_ROOT, "vllm_omni")
_GPU_TRANSPORT_PKG = "vllm_omni.distributed.gpu_transport"
_GPU_TRANSPORT_BASE = f"{_VLLM_OMNI_BASE}/distributed/gpu_transport"


def _setup_vllm_omni_hierarchy():
    """Create stub packages for the vllm_omni hierarchy and load
    gpu_transport submodules via importlib.

    Idempotent -- safe to call multiple times.
    """
    # 1. Stub packages above gpu_transport
    for name, path in [
        ("vllm_omni", _VLLM_OMNI_BASE),
        ("vllm_omni.distributed", f"{_VLLM_OMNI_BASE}/distributed"),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    # 2. Stub for gpu_transport itself
    if _GPU_TRANSPORT_PKG not in sys.modules:
        pkg_mod = types.ModuleType(_GPU_TRANSPORT_PKG)
        pkg_mod.__path__ = [_GPU_TRANSPORT_BASE]
        pkg_mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[_GPU_TRANSPORT_PKG] = pkg_mod

    # 3. Load submodules in dependency order
    def _load(name, file):
        full = f"{_GPU_TRANSPORT_PKG}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        path = f"{_GPU_TRANSPORT_BASE}/{file}"
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    # Layer 1: no intra-package deps
    _load("logging", "logging.py")
    _load("protocol", "protocol.py")
    _load("config", "config.py")

    # Layer 2: depend on logging
    _load("tensor_registry", "tensor_registry.py")
    _load("ipc_utils", "ipc_utils.py")

    # Layer 3: depend on everything above
    _load("cuda_ipc_transport", "cuda_ipc_transport.py")
    _load("cuda_copy_transport", "cuda_copy_transport.py")


_setup_vllm_omni_hierarchy()

# ---------------------------------------------------------------------------
# Imports  (these work because the hierarchy is already set up above)
# ---------------------------------------------------------------------------

from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
from vllm_omni.distributed.gpu_transport.protocol import TensorMetadata, TransportHandle

# Set spawn start method for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consumer helper  (runs in a spawned child process)
# ---------------------------------------------------------------------------


def _consumer_process(conn, mode, src_dev, dst_dev):
    """Run in spawned consumer process.

    Receives metadata + ipc_args over the pipe, reconstructs the tensor
    via the appropriate transport, verifies data integrity by computing
    the sum, and sends an ACK back to the producer.
    """
    # Ensure the hierarchy is set up in the child process too
    _setup_vllm_omni_hierarchy()

    if mode == "cuda_ipc":
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (  # noqa: F811
            CudaIpcTransport,
        )
        transport = CudaIpcTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev, dst_device=dst_dev)
        )
    else:
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (  # noqa: F811
            CudaCopyTransport,
        )
        transport = CudaCopyTransport(
            GPUTransportConfig(mode=mode, src_device=src_dev, dst_device=dst_dev)
        )

    try:
        msg = conn.recv()
        meta_dict = msg["metadata"]
        ipc_args = msg["ipc_args"]
        metadata = TensorMetadata.from_dict(meta_dict)
        metadata.ipc_args = ipc_args
        handle = TransportHandle(tensor_id=metadata.tensor_id, metadata=metadata)

        tensor = transport.recv(
            handle, src_rank=src_dev, dst_device=f"cuda:{dst_dev}"
        )

        result_sum = tensor.sum().item()

        ack_type = "copy_done" if mode == "cuda_copy" else "release"
        conn.send(
            {
                "type": ack_type,
                "tensor_id": metadata.tensor_id,
                "sum": result_sum,
            }
        )
    finally:
        transport.close()
        conn.close()


# ---------------------------------------------------------------------------
# TestDataCorrectness  -- cross-process roundtrip checksum verification
# ---------------------------------------------------------------------------


class TestDataCorrectness:
    """Verify that tensors survive a full send/recv roundtrip across two
    GPUs."""

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_tensor_roundtrip_cuda_ipc(self):
        """Producer sends random tensor via CUDA IPC, consumer receives and
        verifies checksum."""
        src_dev, dst_dev = 0, 1

        _setup_vllm_omni_hierarchy()
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        prod_transport = CudaIpcTransport(
            GPUTransportConfig(
                mode="cuda_ipc", src_device=src_dev, dst_device=dst_dev
            )
        )

        tensor = torch.randn(256, 256, dtype=torch.float32, device=f"cuda:{src_dev}")
        expected_sum = tensor.sum().item()

        handle = prod_transport.send(tensor, dst_rank=dst_dev)

        prod_conn, cons_conn = mp.Pipe()
        proc = mp.Process(
            target=_consumer_process, args=(cons_conn, "cuda_ipc", src_dev, dst_dev)
        )
        proc.start()
        cons_conn.close()

        prod_conn.send(
            {
                "metadata": handle.metadata.to_dict(),
                "ipc_args": handle.metadata.ipc_args,
            }
        )

        ack = prod_conn.recv()
        assert ack["type"] in ("release", "copy_done"), (
            f"Unexpected ACK type: {ack['type']}"
        )
        assert ack["tensor_id"] == handle.tensor_id

        assert abs(ack["sum"] - expected_sum) < 1e-3, (
            f"Checksum mismatch: expected={expected_sum}, got={ack['sum']}"
        )

        prod_transport.release(handle.tensor_id)

        proc.join(timeout=30)
        assert proc.exitcode == 0, (
            f"Consumer process failed with exit code {proc.exitcode}"
        )
        prod_transport.close()
        prod_conn.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_tensor_roundtrip_cuda_copy(self):
        """Producer sends random tensor via CUDA copy, consumer receives and
        verifies checksum."""
        src_dev, dst_dev = 0, 1

        _setup_vllm_omni_hierarchy()
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )

        prod_transport = CudaCopyTransport(
            GPUTransportConfig(
                mode="cuda_copy", src_device=src_dev, dst_device=dst_dev
            )
        )

        tensor = torch.randn(256, 256, dtype=torch.float32, device=f"cuda:{src_dev}")
        expected_sum = tensor.sum().item()

        handle = prod_transport.send(tensor, dst_rank=dst_dev)

        prod_conn, cons_conn = mp.Pipe()
        proc = mp.Process(
            target=_consumer_process,
            args=(cons_conn, "cuda_copy", src_dev, dst_dev),
        )
        proc.start()
        cons_conn.close()

        prod_conn.send(
            {
                "metadata": handle.metadata.to_dict(),
                "ipc_args": handle.metadata.ipc_args,
            }
        )

        ack = prod_conn.recv()
        assert ack["type"] in ("copy_done", "release"), (
            f"Unexpected ACK type: {ack['type']}"
        )
        assert ack["tensor_id"] == handle.tensor_id

        assert abs(ack["sum"] - expected_sum) < 1e-3, (
            f"Checksum mismatch: expected={expected_sum}, got={ack['sum']}"
        )

        prod_transport.release(handle.tensor_id)

        proc.join(timeout=30)
        assert proc.exitcode == 0, (
            f"Consumer process failed with exit code {proc.exitcode}"
        )
        prod_transport.close()
        prod_conn.close()


# ---------------------------------------------------------------------------
# TestLifecycle  -- registry size tracking and unknown-id safety
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Verify registry entry lifecycle: creation, release, and unknown-ID
    handling."""

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_registry_holds_until_release(self):
        """After send(), registry has 1 entry. After release(), registry is
        0."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        tensor = torch.randn(100, device="cuda:0")
        handle = transport.send(tensor, dst_rank=1)

        assert transport.registry_size == 1, (
            f"Expected registry size 1 after send, got {transport.registry_size}"
        )

        transport.release(handle.tensor_id)
        assert transport.registry_size == 0, (
            f"Expected registry size 0 after release, got {transport.registry_size}"
        )

        transport.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_registry_release_unknown(self):
        """release() of unknown id should not raise."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        transport.release("nonexistent-id-12345")
        assert transport.registry_size == 0

        transport.close()


# ---------------------------------------------------------------------------
# TestNoLeak  -- single-process memory stability
# ---------------------------------------------------------------------------


class TestNoLeak:
    """Verify that repeated send/release cycles do not leak GPU memory."""

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_single_process_1000_rounds_no_leak(self):
        """1000 send/release cycles on same process, GPU memory does not
        grow beyond margin."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        torch.cuda.synchronize(0)
        initial_mem = torch.cuda.memory_allocated(0)

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        for _ in range(1000):
            t = torch.randn(1000, device="cuda:0")
            handle = transport.send(t, dst_rank=1)
            transport.release(handle.tensor_id)

        torch.cuda.synchronize(0)
        final_mem = torch.cuda.memory_allocated(0)

        transport.close()

        mem_growth = final_mem - initial_mem
        assert mem_growth < 50_000_000, (
            f"GPU memory grew by {mem_growth / 1e6:.2f} MB (limit 50 MB). "
            f"Initial: {initial_mem / 1e6:.2f} MB, Final: {final_mem / 1e6:.2f} MB"
        )


# ---------------------------------------------------------------------------
# TestConcurrency  -- multi-tensor out-of-order release
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Verify registry correctly handles multiple in-flight tensors."""

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_multi_tensor_out_of_order_release(self):
        """Send 10 tensors, release in reverse order, registry count goes to
        0."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        handles = []
        for _ in range(10):
            t = torch.randn(100, device="cuda:0")
            handle = transport.send(t, dst_rank=1)
            handles.append(handle)

        assert transport.registry_size == 10, (
            f"Expected registry size 10 after 10 sends, got {transport.registry_size}"
        )

        for handle in reversed(handles):
            transport.release(handle.tensor_id)

        assert transport.registry_size == 0, (
            f"Expected registry size 0 after all releases, got {transport.registry_size}"
        )

        transport.close()


# ---------------------------------------------------------------------------
# TestUnsupported  -- error handling for invalid inputs
# ---------------------------------------------------------------------------


class TestUnsupported:
    """Verify correct error handling for invalid operations."""

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_cpu_tensor_raises(self):
        """send(CPU tensor) raises ValueError."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        cpu_tensor = torch.randn(100)
        with pytest.raises(ValueError, match="Only CUDA tensors supported"):
            transport.send(cpu_tensor, dst_rank=1)

        transport.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_wrong_mode_raises(self):
        """recv with wrong mode in metadata raises ValueError."""
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        # CudaIpcTransport recv with mode=cuda_copy
        ipc_transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )
        t = torch.randn(100, device="cuda:0")
        handle = ipc_transport.send(t, dst_rank=1)
        handle.metadata.mode = "cuda_copy"
        with pytest.raises(ValueError, match="Expected mode=cuda_ipc"):
            ipc_transport.recv(handle, src_rank=0, dst_device="cuda:1")
        ipc_transport.release(handle.tensor_id)
        ipc_transport.close()

        # CudaCopyTransport recv with mode=cuda_ipc
        copy_transport = CudaCopyTransport(
            GPUTransportConfig(mode="cuda_copy", src_device=0, dst_device=1)
        )
        t2 = torch.randn(100, device="cuda:0")
        handle2 = copy_transport.send(t2, dst_rank=1)
        handle2.metadata.mode = "cuda_ipc"
        with pytest.raises(ValueError, match="Expected mode=cuda_copy"):
            copy_transport.recv(handle2, src_rank=0, dst_device="cuda:1")
        copy_transport.release(handle2.tensor_id)
        copy_transport.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_non_contiguous_logs_warning(self, caplog):
        """send(non-contiguous tensor) records contiguous=False in metadata."""
        from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import (
            CudaIpcTransport,
        )

        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )

        # Create a non-contiguous tensor via transpose
        t = torch.randn(100, 200, device="cuda:0")
        t_non_contig = t.t()
        assert not t_non_contig.is_contiguous(), "Expected non-contiguous tensor"

        caplog.set_level(logging.WARNING)
        handle = transport.send(t_non_contig, dst_rank=1)

        # The metadata should record that the ORIGINAL tensor was not contiguous
        assert handle.metadata.contiguous is False, (
            f"Expected contiguous=False in metadata for non-contiguous input, "
            f"got {handle.metadata.contiguous}"
        )

        transport.release(handle.tensor_id)
        transport.close()
