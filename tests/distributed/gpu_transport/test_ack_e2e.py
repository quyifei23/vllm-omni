# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end ACK flow tests for GPU transport.

Verifies that the producer's ACK daemon thread automatically releases a tensor
when it receives a copy_done ACK from the consumer.
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

pytestmark = [pytest.mark.gpu]


def _consumer_process(meta_conn, ack_conn, src_dev, dst_dev):
    """Consumer that uses consumer_ack_conn to send copy_done ACK.

    Runs in a spawned child process.
    """
    try:
        _setup_vllm_omni_hierarchy()
        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )
        from vllm_omni.distributed.gpu_transport.protocol import (
            TensorMetadata,
            TransportHandle,
        )

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


class TestAckE2E:

    def test_cuda_copy_ack_releases_automatically(self):
        """ACK thread on producer auto-releases tensor on copy_done ACK."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        meta_prod, meta_cons = mp.Pipe()
        ack_prod, ack_cons = mp.Pipe()
        src_dev, dst_dev = 0, 0

        proc = mp.Process(
            target=_consumer_process,
            args=(meta_cons, ack_cons, src_dev, dst_dev),
        )
        proc.start()
        meta_cons.close()
        ack_cons.close()

        from vllm_omni.distributed.gpu_transport.cuda_copy_transport import (
            CudaCopyTransport,
        )

        # Producer WITH ack_conn (starts ACK thread)
        producer = CudaCopyTransport(GPUTransportConfig(
            mode="cuda_copy", src_device=src_dev, dst_device=dst_dev,
            ack_conn=ack_prod,
        ))

        try:
            tensor = torch.ones(8, dtype=torch.float32, device=f"cuda:{src_dev}")
            expected_sum = tensor.sum().item()
            handle = producer.send(tensor, dst_rank=dst_dev)
            assert producer.registry_size == 1, (
                "Tensor should be in registry after send"
            )

            # Send metadata to consumer
            meta_prod.send({
                "metadata": handle.metadata.to_dict(),
                "ipc_args": handle.metadata.ipc_args,
            })

            # Wait for consumer result (data integrity check)
            result = meta_prod.recv()
            if result["type"] == "error":
                proc.join(timeout=5)
                raise RuntimeError(
                    f"Consumer failed (exit={proc.exitcode}): {result['msg']}"
                )
            assert result["type"] == "result"
            assert abs(result["sum"] - expected_sum) < 1e-3

            # Consumer has received the tensor and sent copy_done ACK.
            # Producer's ACK thread should auto-release the tensor.
            for _ in range(10):  # up to 5 seconds (ACK thread polls every 500ms)
                if producer.registry_size == 0:
                    break
                time.sleep(0.5)

            assert producer.registry_size == 0, (
                "ACK thread should auto-release tensor after receiving copy_done"
            )
        finally:
            producer.close()
            meta_prod.close()
            ack_prod.close()

        proc.join(timeout=30)
        assert proc.exitcode == 0, (
            f"Consumer process failed with exit code {proc.exitcode}"
        )
