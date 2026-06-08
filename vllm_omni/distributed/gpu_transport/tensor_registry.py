# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from typing import Any

import torch

from .logging import get_logger

logger = get_logger(__name__)


class TensorRegistry:
    """Producer-side: tensor_id -> retained tensor tracking.

    Thread-safe. Uses ``time.monotonic()`` for timeout calculations
    so NTP adjustments and wall-clock changes do not perturb cleanup.
    """

    def __init__(self, timeout_ms: float = 10_000.0):
        self._entries: dict[str, dict[str, Any]] = {}
        self._timeout_ms = timeout_ms
        self._lock = threading.Lock()

    def register(self, tensor_id: str, tensor: torch.Tensor) -> None:
        with self._lock:
            if tensor_id in self._entries:
                logger.warning("registry: overwriting existing entry id=%s", tensor_id)
            nbytes = tensor.numel() * tensor.element_size()
            self._entries[tensor_id] = {
                "tensor": tensor,
                "create_time": time.monotonic(),
                "nbytes": nbytes,
            }
            total = sum(e["nbytes"] for e in self._entries.values())
            count = len(self._entries)
            logger.debug("registry: +%s nbytes=%d size=%d total_bytes=%d",
                         tensor_id, nbytes, count, total)
            # Log potential OOM every 512MB accumulated or at size milestones
            if total > 512 * 1024 * 1024 and total % (512 * 1024 * 1024) < nbytes:
                logger.warning(
                    "registry: HIGH_MEMORY total_bytes=%d size=%d "
                    "— GPU OOM risk if consumer slow to ACK",
                    total, count)

    def release(self, tensor_id: str) -> bool:
        """Returns True if the entry existed and was removed."""
        with self._lock:
            if tensor_id not in self._entries:
                logger.warning("registry: release of unknown id=%s", tensor_id)
                return False
            del self._entries[tensor_id]
            remaining = len(self._entries)
        logger.debug("registry: -%s remaining=%d", tensor_id, remaining)
        return True

    def exists(self, tensor_id: str) -> bool:
        with self._lock:
            return tensor_id in self._entries

    def cleanup_timeouts(self) -> list[str]:
        # timeout_ms <= 0 disables cleanup (baseline / OOM testing)
        if self._timeout_ms <= 0:
            return []
        now = time.monotonic()
        timeout_s = self._timeout_ms / 1000.0
        with self._lock:
            stale = [tid for tid, e in self._entries.items()
                     if (now - e["create_time"]) > timeout_s]
            for tid in stale:
                logger.warning("registry: timeout cleanup id=%s", tid)
                del self._entries[tid]
        return stale

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
