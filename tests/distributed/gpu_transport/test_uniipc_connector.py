"""Integration tests for UniIPCConnector."""
from __future__ import annotations

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs for GPU transport")]


class TestUniIPCConnector:
    def test_put_get_without_gpu_tensor(self):
        """Pure metadata payload passes through unchanged."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        config = {
            "stage_id": 0,
            "device": "cuda:0",
            "shm_threshold_bytes": 65536,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        }
        connector = UniIPCConnector(config)
        data = {"a": 1, "b": "hello", "c": [1, 2, 3]}

        success, size, metadata = connector.put("0", "1", "test-cpu", data)
        assert success is True
        assert size > 0

        result = connector.get("0", "1", "test-cpu", metadata)
        assert result is not None
        obj, _ = result
        assert obj == data

        connector.cleanup("test-cpu")
        connector.close()

    def test_factory_creates_connector(self):
        """OmniConnectorFactory can create UniIPCConnector from spec."""
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        spec = ConnectorSpec(
            name="UniIPCConnector",
            extra={
                "stage_id": 0,
                "gpu_transport_mode": "cuda_ipc",
                "src_device": 0,
                "dst_device": 1,
            },
        )
        connector = OmniConnectorFactory.create_connector(spec)
        assert isinstance(connector, UniIPCConnector)
        assert connector._transport_mode == "cuda_ipc"
        connector.close()

    def test_close_idempotent(self):
        """close() is safe to call multiple times."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        })
        connector.close()
        connector.close()  # second close should not raise

    def test_health_reports_metrics(self):
        """health() returns expected fields."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_copy",
            "src_device": 0,
            "dst_device": 1,
        })
        h = connector.health()
        assert h["status"] == "healthy"
        assert h["transport_mode"] == "cuda_copy"
        assert "puts" in h
        assert "gets" in h
        assert "gpu_tensors_sent" in h
        assert "gpu_tensors_recv" in h
        connector.close()

    def test_factory_list_includes_connector(self):
        """UniIPCConnector is in the factory registry."""
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        connectors = OmniConnectorFactory.list_registered_connectors()
        assert "UniIPCConnector" in connectors


class TestUniIPCConnectorConfig:
    """Tests for new configuration keys."""

    def test_default_min_bytes(self):
        """gpu_transport_min_bytes defaults to 65536."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        })
        assert connector._gpu_transport_min_bytes == 65536
        connector.close()

    def test_custom_min_bytes(self):
        """gpu_transport_min_bytes can be set via config."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 1024,
        })
        assert connector._gpu_transport_min_bytes == 1024
        connector.close()

    def test_default_pressure_threshold(self):
        """gpu_memory_pressure_threshold defaults to 0.0 (disabled)."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        })
        assert connector._gpu_memory_pressure_threshold == 0.0
        connector.close()

    def test_custom_pressure_threshold(self):
        """gpu_memory_pressure_threshold can be set via config."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_memory_pressure_threshold": 0.5,
        })
        assert connector._gpu_memory_pressure_threshold == 0.5
        connector.close()

    def test_mode_none_parsed(self):
        """gpu_transport_mode='none' is accepted."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "none",
            "src_device": 0,
            "dst_device": 1,
        })
        assert connector._transport_mode == "none"
        connector.close()


class TestUniIPCConnectorRouter:
    """Tests for _route_tensor decisions."""

    def test_router_small_tensor_goes_inline(self):
        """Tensor below min_bytes returns 'inline'."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 1024,
        })
        t = torch.ones(10, device="cuda:0", dtype=torch.float32)  # 40 bytes
        assert connector._route_tensor(t) == "inline"
        connector.close()

    def test_router_large_tensor_goes_ipc(self):
        """Tensor above min_bytes returns 'ipc'."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 100,
        })
        t = torch.ones(100, device="cuda:0", dtype=torch.float32)  # 400 bytes
        assert connector._route_tensor(t) == "ipc"
        connector.close()

    def test_router_mode_none_always_inline(self):
        """Mode 'none' returns 'inline' regardless of size."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "none",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 100,
        })
        t = torch.ones(1000, device="cuda:0", dtype=torch.float32)  # 4000 bytes
        assert connector._route_tensor(t) == "inline"
        connector.close()

    def test_router_pressure_fallback(self):
        """When free memory falls below threshold, returns 'inline'."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 100,
            "gpu_memory_pressure_threshold": 0.999,  # impossibly high
        })
        t = torch.ones(1000, device="cuda:0", dtype=torch.float32)
        decision = connector._route_tensor(t)
        # With threshold=0.999 essentially all free ratios are below it
        assert decision == "inline"
        assert connector._metrics["pressure_fallbacks"] >= 1
        connector.close()

    def test_router_pressure_disabled(self):
        """With threshold=0.0, pressure check is skipped."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 100,
            "gpu_memory_pressure_threshold": 0.0,
        })
        t = torch.ones(1000, device="cuda:0", dtype=torch.float32)
        assert connector._route_tensor(t) == "ipc"
        connector.close()


class TestUniIPCConnectorInline:
    """End-to-end tests for the inline path."""

    def test_put_get_small_gpu_tensor_inlined(self):
        """Small GPU tensor goes inline, roundtrips correctly."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "device": "cuda:0",
            "shm_threshold_bytes": 65536,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 10000,  # force inline for small tensor
        })
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        data = {"tensor": t, "scalar": 42}

        success, size, metadata = connector.put("0", "1", "test-inline", data)
        assert success is True

        result = connector.get("0", "1", "test-inline", metadata)
        assert result is not None
        obj, _ = result
        assert "tensor" in obj
        assert obj["scalar"] == 42
        assert torch.allclose(obj["tensor"].cpu(), t.cpu())
        assert connector._metrics["gpu_tensors_inlined"] >= 1

        connector.cleanup("test-inline")
        connector.close()

    def test_put_get_mixed_inline_tensors(self):
        """Multiple tensors in payload both go inline, roundtrip correctly."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "device": "cuda:0",
            "shm_threshold_bytes": 65536,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            # threshold above both tensor sizes so both go inline
            "gpu_transport_min_bytes": 1000,
        })
        t_small = torch.tensor([1.0], device="cuda:0")       # 4 bytes
        t_large = torch.randn(100, device="cuda:0", dtype=torch.float32)  # 400 bytes
        data = {"small": t_small, "large": t_large}

        success, size, metadata = connector.put("0", "1", "test-mixed", data)
        assert success is True

        result = connector.get("0", "1", "test-mixed", metadata)
        assert result is not None
        obj, _ = result
        assert torch.allclose(obj["small"].cpu(), t_small.cpu())
        assert torch.allclose(obj["large"].cpu(), t_large.cpu())
        assert connector._metrics["gpu_tensors_inlined"] >= 2

        connector.cleanup("test-mixed")
        connector.close()

    def test_mode_none_puts_gpu_tensor_inline(self):
        """Mode 'none' sends GPU tensors as inline bytes successfully."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        import torch
        connector = UniIPCConnector({
            "stage_id": 0,
            "device": "cuda:0",
            "shm_threshold_bytes": 65536,
            "gpu_transport_mode": "none",
            "src_device": 0,
            "dst_device": 1,
        })
        t = torch.randn(50, device="cuda:0")
        data = {"tensor": t}

        success, size, metadata = connector.put("0", "1", "test-none", data)
        assert success is True

        result = connector.get("0", "1", "test-none", metadata)
        assert result is not None
        obj, _ = result
        assert torch.allclose(obj["tensor"].cpu(), t.cpu())

        connector.cleanup("test-none")
        connector.close()

    def test_health_reports_new_metrics(self):
        """health() includes new config and metric fields."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
            "gpu_transport_min_bytes": 4096,
            "gpu_memory_pressure_threshold": 0.8,
        })
        h = connector.health()
        assert h["gpu_transport_min_bytes"] == 4096
        assert h["gpu_memory_pressure_threshold"] == 0.8
        assert "gpu_tensors_inlined" in h
        assert "inline_bytes" in h
        assert "pressure_fallbacks" in h
        assert "current_registry_size" in h
        assert "gpu_free_memory_ratio" in h
        connector.close()
