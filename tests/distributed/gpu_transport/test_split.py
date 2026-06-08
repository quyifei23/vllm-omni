# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for payload split/reassemble of GPU tensor transport.

Verifies that ``has_gpu_tensors``, ``split_gpu_tensors``, and
``reassemble_gpu_tensors`` work correctly.

Single-process tests cover marker creation and detection.
Cross-process (CUDA IPC) tests cover the full send/reassemble roundtrip
since ``rebuild_cuda_tensor`` requires a separate consumer process.
"""
from __future__ import annotations

import importlib.util
import io
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
    _load("control_channel", "control_channel.py")

    # Layer 3: depend on everything above
    _load("split", "split.py")
    _load("cuda_ipc_transport", "cuda_ipc_transport.py")
    _load("cuda_copy_transport", "cuda_copy_transport.py")


_setup_vllm_omni_hierarchy()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from vllm_omni.distributed.gpu_transport.split import (
    has_gpu_tensors,
    reassemble_gpu_tensors,
    split_gpu_tensors,
)
from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import CudaIpcTransport

# Set spawn start method for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

pytestmark = [pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Consumer helper  (runs in a spawned child process)
# ---------------------------------------------------------------------------


def _reassemble_consumer(conn, mode, src_dev, dst_dev):
    """Run in spawned consumer process.

    Receives a marker dict (produced by split_gpu_tensors), calls
    reassemble_gpu_tensors to restore GPU tensors, computes a checksum,
    and sends the result back to the producer.
    """
    _setup_vllm_omni_hierarchy()
    from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
    from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import CudaIpcTransport
    from vllm_omni.distributed.gpu_transport.split import reassemble_gpu_tensors
    import torch
    transport = CudaIpcTransport(
        GPUTransportConfig(mode=mode, src_device=src_dev, dst_device=dst_dev)
    )
    try:
        payload = conn.recv()
        restored = reassemble_gpu_tensors(payload, transport)

        # Collect checksums for all GPU tensors in the structure
        def _walk_and_sum(obj):
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                return obj.sum().item()
            if isinstance(obj, dict):
                return {k: _walk_and_sum(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_walk_and_sum(v) for v in obj]
            return obj

        result = _walk_and_sum(restored)
        conn.send({"type": "result", "data": result})
    finally:
        transport.close()
        conn.close()


class TestHasGpuTensors:
    def test_no_gpu_tensor(self):
        assert not has_gpu_tensors({"a": 1, "b": "hello"})
        assert not has_gpu_tensors([1, 2, 3])
        assert not has_gpu_tensors("string")

    def test_gpu_tensor_detected(self):
        assert has_gpu_tensors(torch.ones(10, device="cuda:0"))

    def test_nested_gpu_tensor(self):
        assert has_gpu_tensors({"a": {"b": torch.ones(10, device="cuda:0")}})
        assert has_gpu_tensors([1, [2, torch.ones(5, device="cuda:0")]])


class TestSplitReassemble:
    def test_no_gpu_tensor_unchanged(self):
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))
        obj = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        result = split_gpu_tensors(obj, transport)
        assert result == obj
        transport.close()

    def test_split_creates_markers(self):
        """Verify that split_gpu_tensors creates gpux markers correctly
        (single-process test, no reassembly needed)."""
        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=0, dst_device=1)
        )
        t = torch.randn(10, device="cuda:0", dtype=torch.float16)
        payload = {"tensor": t}

        split_result = split_gpu_tensors(payload, transport)
        assert not has_gpu_tensors(split_result)
        assert split_result["tensor"]["__gpux__"] is True
        assert "tensor_id" in split_result["tensor"]
        assert "meta" in split_result["tensor"]
        assert "ipc_args" in split_result["tensor"]

        transport.release(split_result["tensor"]["tensor_id"])
        transport.close()

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_single_tensor_roundtrip(self):
        """Producer splits, consumer reassembles via CUDA IPC."""
        src_dev, dst_dev = 0, 1

        prod_transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=src_dev, dst_device=dst_dev)
        )

        t = torch.randn(10, device="cuda:0", dtype=torch.float16)
        payload = {"tensor": t}
        split_result = split_gpu_tensors(payload, prod_transport)

        prod_conn, cons_conn = mp.Pipe()
        proc = mp.Process(
            target=_reassemble_consumer,
            args=(cons_conn, "cuda_ipc", src_dev, dst_dev),
        )
        proc.start()
        cons_conn.close()

        prod_conn.send(split_result)
        ack = prod_conn.recv()
        assert ack["type"] == "result"
        assert abs(ack["data"]["tensor"] - t.sum().item()) < 1e-3, (
            f"Checksum mismatch: expected={t.sum().item()}, got={ack['data']}"
        )

        prod_transport.release(split_result["tensor"]["tensor_id"])

        proc.join(timeout=30)
        assert proc.exitcode == 0, (
            f"Consumer failed with exit code {proc.exitcode}"
        )
        prod_transport.close()
        prod_conn.close()

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need at least 2 GPUs"
    )
    def test_multi_tensor_roundtrip(self):
        """Producer splits a nested payload with multiple tensors,
        consumer reassembles all tensors via CUDA IPC."""
        src_dev, dst_dev = 0, 1

        prod_transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", src_device=src_dev, dst_device=dst_dev)
        )

        t1 = torch.ones(5, device="cuda:0")
        t2 = torch.ones(5, device="cuda:0") * 2
        payload = {"a": {"b": t1, "c": [t2, "not_a_tensor"]}}
        split_result = split_gpu_tensors(payload, prod_transport)
        assert not has_gpu_tensors(split_result)

        prod_conn, cons_conn = mp.Pipe()
        proc = mp.Process(
            target=_reassemble_consumer,
            args=(cons_conn, "cuda_ipc", src_dev, dst_dev),
        )
        proc.start()
        cons_conn.close()

        prod_conn.send(split_result)
        ack = prod_conn.recv()
        assert ack["type"] == "result"
        result = ack["data"]
        assert result["a"]["b"] == 5.0
        assert result["a"]["c"][0] == 10.0
        assert result["a"]["c"][1] == "not_a_tensor"

        for tid in [
            split_result["a"]["b"]["tensor_id"],
            split_result["a"]["c"][0]["tensor_id"],
        ]:
            prod_transport.release(tid)

        proc.join(timeout=30)
        assert proc.exitcode == 0, (
            f"Consumer failed with exit code {proc.exitcode}"
        )
        prod_transport.close()
        prod_conn.close()


class TestInlineMarker:
    """Tests for inline (CPU bytes) marker path -- no transport needed."""

    def test_inline_marker_roundtrip_float32(self):
        """Inline marker can roundtrip a float32 tensor correctly."""
        from vllm_omni.distributed.gpu_transport.split import (
            _make_inline_marker,
            _recover_inline_tensor,
        )
        t = torch.randn(3, 4, device="cuda:0", dtype=torch.float32)
        marker = _make_inline_marker(t, dst_device="cuda:0")
        assert marker["__gpux__"] is True
        assert "inline_data" in marker
        assert "ipc_args" not in marker

        restored = _recover_inline_tensor(marker)
        assert restored.device.type == "cuda"
        assert restored.shape == t.shape
        assert restored.dtype == t.dtype
        assert torch.allclose(restored, t)

    def test_inline_marker_roundtrip_float16(self):
        """Inline marker handles float16 correctly."""
        from vllm_omni.distributed.gpu_transport.split import (
            _make_inline_marker,
            _recover_inline_tensor,
        )
        t = torch.randn(2, 8, device="cuda:0", dtype=torch.float16)
        marker = _make_inline_marker(t, dst_device="cuda:0")
        restored = _recover_inline_tensor(marker)
        assert torch.allclose(restored, t)

    def test_inline_marker_roundtrip_bfloat16(self):
        """Inline marker handles bfloat16 correctly."""
        from vllm_omni.distributed.gpu_transport.split import (
            _make_inline_marker,
            _recover_inline_tensor,
        )
        t = torch.randn(4, device="cuda:0", dtype=torch.bfloat16)
        marker = _make_inline_marker(t, dst_device="cuda:0")
        restored = _recover_inline_tensor(marker)
        assert torch.allclose(restored, t)

    def test_inline_marker_roundtrip_int64(self):
        """Inline marker handles integer tensors."""
        from vllm_omni.distributed.gpu_transport.split import (
            _make_inline_marker,
            _recover_inline_tensor,
        )
        t = torch.tensor([1, 2, 3], device="cuda:0", dtype=torch.int64)
        marker = _make_inline_marker(t, dst_device="cuda:0")
        restored = _recover_inline_tensor(marker)
        assert torch.equal(restored, t)

    def test_inline_marker_non_contiguous(self):
        """Inline marker handles non-contiguous tensors."""
        from vllm_omni.distributed.gpu_transport.split import (
            _make_inline_marker,
            _recover_inline_tensor,
        )
        t = torch.randn(4, 6, device="cuda:0")
        sliced = t[:, ::2]  # non-contiguous
        assert not sliced.is_contiguous()
        marker = _make_inline_marker(sliced, dst_device="cuda:0")
        restored = _recover_inline_tensor(marker)
        assert torch.allclose(restored, sliced)
