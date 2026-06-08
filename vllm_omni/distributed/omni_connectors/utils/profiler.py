# SPDX-License-Identifier: Apache-2.0
"""Lightweight profiler for OmniConnector communication.

Controlled by environment variables:
  VLLM_OMNI_CONNECTOR_PROFILING=1   -- enable profiling
  VLLM_OMNI_CONNECTOR_PROFILING_OUTPUT=<path>  -- output JSONL file (default: auto-generated)

Design constraints:
  - Near-zero overhead when disabled (_enabled short-circuit).
  - Enabled path writes JSONL asynchronously to avoid blocking data plane.
  - Multi-process safe: each PID writes to a separate file.
  - Never triggers tensor copies, GPU syncs, or extra data movement.
  - Flushes on process exit via atexit.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import deque
from typing import Any

_JSON_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    default=lambda o: f"<{type(o).__name__}>",
)


class OmniConnectorProfiler:
    """Singleton profiler for OmniConnector communication events."""

    _instance: OmniConnectorProfiler | None = None
    _lock = threading.Lock()

    def __new__(cls) -> OmniConnectorProfiler:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._enabled: bool = bool(int(os.environ.get("VLLM_OMNI_CONNECTOR_PROFILING", "0")))
        self._output_path: str = os.environ.get(
            "VLLM_OMNI_CONNECTOR_PROFILING_OUTPUT",
            f"/tmp/omni_connector_trace_{os.getpid()}.jsonl",
        )
        self._buffer: deque[dict] = deque()
        self._file: Any = None
        self._write_lock = threading.Lock()
        self._flush_timer: threading.Timer | None = None
        self._flush_interval: float = 0.5  # seconds
        self._flushed: bool = False

        if self._enabled:
            self._open_file()
            atexit.register(self._close)

    def _open_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
            self._file = open(self._output_path, "a", buffering=1)
        except Exception:
            self._file = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._file is not None

    def record_event(self, **kwargs: Any) -> None:
        """Record a profiler event. Non-blocking; events are buffered and flushed periodically.

        Keys that will be added automatically:
          timestamp_ns, pid
        """
        if not self.enabled:
            return
        kwargs.setdefault("timestamp_ns", time.time_ns())
        kwargs.setdefault("pid", os.getpid())
        kwargs.setdefault("thread_id", threading.get_ident())
        with self._write_lock:
            self._buffer.append(kwargs)
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_timer is not None:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self._do_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _do_flush(self) -> None:
        self._flush_timer = None
        self.flush()

    def flush(self) -> None:
        """Flush buffered events to disk. Keeps the file open for future writes."""
        with self._write_lock:
            if self._buffer and self._file is not None:
                try:
                    for event in self._buffer:
                        self._file.write(_JSON_ENCODER.encode(event) + "\n")
                    self._file.flush()
                except Exception:
                    pass
                self._buffer.clear()

    def _close(self) -> None:
        """Close the output file. Called by atexit only."""
        if self._flushed:
            return
        self._flushed = True
        self.flush()
        with self._write_lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None


def get_connector_profiler() -> OmniConnectorProfiler:
    """Get the global profiler singleton."""
    return OmniConnectorProfiler()


def profiled_put(connector, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
    """Wrapper around connector.put() with profiling."""
    profiler = get_connector_profiler()
    t0 = time.perf_counter_ns()

    from vllm_omni.distributed.omni_connectors.utils.payload_inspector import inspect_payload

    info = inspect_payload(data)

    try:
        success, size, metadata = connector.put(from_stage, to_stage, put_key, data)
    except Exception as e:
        t1 = time.perf_counter_ns()
        profiler.record_event(
            event_type="connector_put",
            request_id=put_key,
            from_stage=from_stage,
            to_stage=to_stage,
            connector_name=type(connector).__name__,
            put_time_us=(t1 - t0) / 1000,
            success=False,
            error_type=type(e).__name__,
            **info,
        )
        raise

    t1 = time.perf_counter_ns()
    is_fast_path = isinstance(metadata, dict) and metadata.get("is_fast_path", False)

    profiler.record_event(
        event_type="connector_put",
        request_id=put_key,
        from_stage=from_stage,
        to_stage=to_stage,
        connector_name=type(connector).__name__,
        put_time_us=(t1 - t0) / 1000,
        metadata_size_bytes=size,
        is_fast_path=is_fast_path,
        success=success,
        **info,
    )
    return success, size, metadata


def profiled_get(connector, from_stage: str, to_stage: str, get_key: str, metadata=None) -> Any:
    """Wrapper around connector.get() with profiling."""
    profiler = get_connector_profiler()
    t0 = time.perf_counter_ns()

    try:
        result = connector.get(from_stage, to_stage, get_key, metadata)
    except Exception as e:
        t1 = time.perf_counter_ns()
        profiler.record_event(
            event_type="connector_get",
            request_id=get_key,
            from_stage=from_stage,
            to_stage=to_stage,
            connector_name=type(connector).__name__,
            get_time_us=(t1 - t0) / 1000,
            success=False,
            error_type=type(e).__name__,
        )
        raise

    t1 = time.perf_counter_ns()
    if result is not None:
        data, size = result
        from vllm_omni.distributed.omni_connectors.utils.payload_inspector import inspect_payload

        info = inspect_payload(data)
    else:
        info = {}

    profiler.record_event(
        event_type="connector_get",
        request_id=get_key,
        from_stage=from_stage,
        to_stage=to_stage,
        connector_name=type(connector).__name__,
        get_time_us=(t1 - t0) / 1000,
        success=result is not None,
        **info,
    )
    return result


def profiled_serialize(obj: Any) -> bytes:
    """Wrapper around OmniSerializer.serialize() with profiling."""
    profiler = get_connector_profiler()
    t0 = time.perf_counter_ns()

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    result = OmniSerializer.serialize(obj)

    t1 = time.perf_counter_ns()

    from vllm_omni.distributed.omni_connectors.utils.payload_inspector import inspect_payload

    info = inspect_payload(obj)

    profiler.record_event(
        event_type="serialize",
        serialize_time_us=(t1 - t0) / 1000,
        output_size_bytes=len(result),
        **info,
    )
    return result


def profiled_deserialize(data: bytes) -> Any:
    """Wrapper around OmniSerializer.deserialize() with profiling."""
    profiler = get_connector_profiler()
    t0 = time.perf_counter_ns()

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    result = OmniSerializer.deserialize(data)

    t1 = time.perf_counter_ns()

    from vllm_omni.distributed.omni_connectors.utils.payload_inspector import inspect_payload

    info = inspect_payload(result)

    profiler.record_event(
        event_type="deserialize",
        deserialize_time_us=(t1 - t0) / 1000,
        input_size_bytes=len(data) if isinstance(data, (bytes, bytearray, memoryview)) else 0,
        **info,
    )
    return result
