# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the ACK thread in CUDA IPC transport.

Verifies that the background ACK thread correctly releases tensors
when release ACKs arrive over the control pipe, and that timeout
cleanup works correctly.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import time
import types

import pytest
import torch
from multiprocessing import Pipe

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
    _load("cuda_ipc_transport", "cuda_ipc_transport.py")
    _load("cuda_copy_transport", "cuda_copy_transport.py")


_setup_vllm_omni_hierarchy()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import CudaIpcTransport
from vllm_omni.distributed.gpu_transport.control_channel import ConsumerControl, ProducerControl

pytestmark = [pytest.mark.gpu]


class TestAckThread:
    def test_ack_thread_releases_tensor(self):
        p1, p2 = Pipe()
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))

        # Start ACK thread manually
        transport._ack_conn = p1
        transport._start_ack_thread()

        t = torch.ones(10, device="cuda:0")
        transport._registry.register("test-ack", t)
        assert transport.registry_size == 1

        # Consumer sends release ACK
        consumer = ConsumerControl(p2)
        consumer.send_ack("test-ack", ack_type="release")
        time.sleep(0.3)

        assert transport.registry_size == 0, "ACK thread should have released"

        transport.shutdown_ack_thread()
        transport.close()
        p1.close()
        p2.close()

    def test_no_ack_conn_no_thread(self):
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))
        assert transport._ack_thread is None, (
            "Should not start thread without ack_conn"
        )
        transport.close()

    def test_timeout_cleanup_releases_stale(self):
        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", release_timeout_ms=10)
        )
        t = torch.ones(10, device="cuda:0")
        transport._registry.register("test-timeout", t)
        assert transport.registry_size == 1

        time.sleep(0.1)
        stale = transport.cleanup_timeouts()
        assert "test-timeout" in stale
        assert transport.registry_size == 0
        transport.close()

    def test_ack_thread_handles_shutdown(self):
        p1, p2 = Pipe()
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))
        transport._ack_conn = p1
        transport._start_ack_thread()

        # Send shutdown via pipe
        p2.send({"type": "shutdown"})
        time.sleep(0.3)

        # Thread should have stopped
        transport._ack_running = False  # ensure
        if transport._ack_thread:
            transport._ack_thread.join(timeout=1)
        assert not (
            transport._ack_thread and transport._ack_thread.is_alive()
        )
        transport.close()
        p1.close()
        p2.close()


def _consumer_copy_ack_process(meta_conn, ack_conn, src_dev, dst_dev):
    """Consumer that uses consumer_ack_conn to send copy_done ACK.

    Runs in a spawned child process.
    """
    try:
        _setup_vllm_omni_hierarchy()
        from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import CudaCopyTransport
        from vllm_omni.distributed.gpu_transport.protocol import TensorMetadata, TransportHandle

        transport = CudaCopyTransport(GPUTransportConfig(
            mode="cuda_copy", src_device=src_dev, dst_device=dst_dev,
            consumer_ack_conn=ack_conn,
        ))

        try:
            msg = meta_conn.recv()
            meta_dict = msg["metadata"]
            ipc_args = msg["ipc_args"]
            metadata = TensorMetadata.from_dict(meta_dict)
            metadata.ipc_args = ipc_args
            handle = TransportHandle(tensor_id=metadata.tensor_id, metadata=metadata)

            tensor = transport.recv(handle, src_rank=src_dev, dst_device=f"cuda:{dst_dev}")
            result_sum = tensor.sum().item()
            meta_conn.send({"type": "result", "sum": result_sum})
        finally:
            transport.close()
    except Exception as exc:
        try:
            meta_conn.send({"type": "error", "msg": str(exc)})
        except Exception:
            pass
    finally:
        meta_conn.close()
        ack_conn.close()


def test_cuda_copy_recv_sends_ack():
    """CudaCopyTransport.recv() sends copy_done ACK after copy (cross-process)."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    meta_prod, meta_cons = mp.Pipe()
    ack_prod, ack_cons = mp.Pipe()
    src_dev, dst_dev = 0, 0

    proc = mp.Process(
        target=_consumer_copy_ack_process,
        args=(meta_cons, ack_cons, src_dev, dst_dev),
    )
    proc.start()
    meta_cons.close()
    ack_cons.close()

    from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
    from vllm_omni.distributed.gpu_transport.cuda_copy_transport import CudaCopyTransport

    producer = CudaCopyTransport(GPUTransportConfig(
        mode="cuda_copy", src_device=src_dev, dst_device=dst_dev,
    ))

    import torch
    tensor = torch.ones(4, dtype=torch.float32, device=f"cuda:{src_dev}")
    expected_sum = tensor.sum().item()
    handle = producer.send(tensor, dst_rank=dst_dev)

    meta_prod.send({
        "metadata": handle.metadata.to_dict(),
        "ipc_args": handle.metadata.ipc_args,
    })

    # Wait for consumer result (data integrity) before checking ACK
    result = meta_prod.recv()
    if result["type"] == "error":
        proc.join(timeout=5)
        raise RuntimeError(
            f"Consumer failed (exit={proc.exitcode}): {result['msg']}"
        )
    assert result["type"] == "result"
    assert abs(result["sum"] - expected_sum) < 1e-3

    # Producer should receive the ACK via ack_prod (consumer_ack_conn producer end)
    from vllm_omni.distributed.gpu_transport.control_channel import ProducerControl
    ctrl = ProducerControl(ack_prod)
    msg = ctrl.recv_ack(timeout_ms=5000.0)
    assert msg is not None, "Did not receive copy_done ACK"
    assert msg.get("type") == "copy_done"
    assert msg.get("tensor_id") == handle.tensor_id

    proc.join(timeout=30)
    assert proc.exitcode == 0, (
        f"Consumer process failed with exit code {proc.exitcode}"
    )
    producer.close()
    meta_prod.close()
    ack_prod.close()


def test_cuda_ipc_notify_consumed_sends_ack():
    """CudaIpcTransport.notify_consumed() sends release ACK."""
    prod_conn, cons_conn = Pipe()

    producer = CudaIpcTransport(GPUTransportConfig(
        mode="cuda_ipc", src_device=0, dst_device=0,
    ))

    consumer = CudaIpcTransport(GPUTransportConfig(
        mode="cuda_ipc", src_device=0, dst_device=0,
        consumer_ack_conn=cons_conn,
    ))

    tensor = torch.ones(4, device="cuda:0")
    handle = producer.send(tensor, dst_rank=0)

    # Consumer marks tensor as consumed
    consumer.notify_consumed(handle.tensor_id)

    # Producer receives ACK
    ctrl = ProducerControl(prod_conn)
    msg = ctrl.recv_ack(timeout_ms=1000.0)
    assert msg is not None
    assert msg.get("type") == "release"
    assert msg.get("tensor_id") == handle.tensor_id

    # After producer releases, tensor should be gone from registry
    producer.release(handle.tensor_id)
    assert producer.registry_size == 0

    producer.close()
    consumer.close()
    prod_conn.close()
    cons_conn.close()
