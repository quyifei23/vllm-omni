# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multiprocessing Pipe-based control channel for metadata and ACK exchange.

Carries lightweight metadata dicts and ACK messages — never tensor data.
"""
from __future__ import annotations

import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)


class ControlChannelPair:
    """Creates a producer-side writer and consumer-side reader connected by Pipe."""

    def __init__(self):
        self._prod_conn, self._cons_conn = Pipe()

    @property
    def producer_conn(self) -> Connection:
        return self._prod_conn

    @property
    def consumer_conn(self) -> Connection:
        return self._cons_conn

    def close(self) -> None:
        for c in (self._prod_conn, self._cons_conn):
            try:
                c.close()
            except Exception:
                pass


class ProducerControl:
    """Producer side: sends metadata, receives ACKs."""

    def __init__(self, conn: Connection):
        self._conn = conn

    def send_metadata(self, metadata: dict[str, Any], ipc_args: tuple) -> None:
        msg = {"type": "metadata", "metadata": metadata, "ipc_args": ipc_args}
        self._conn.send(msg)
        logger.debug("control: sent metadata for id=%s", metadata.get("tensor_id"))

    def recv_ack(self, timeout_ms: float | None = None) -> dict[str, Any] | None:
        """Receive an ACK. Returns None on timeout."""
        timeout_s = timeout_ms / 1000.0 if timeout_ms else None
        if not self._conn.poll(timeout_s):
            return None
        try:
            msg = self._conn.recv()
            if isinstance(msg, dict) and msg.get("type") in ("ack", "copy_done", "release"):
                logger.debug("control: recv %s for id=%s",
                            msg.get("type"), msg.get("tensor_id"))
                return msg
        except EOFError:
            logger.warning("control: pipe closed (consumer exited?)")
        return None

    def close(self) -> None:
        self._conn.close()


class ConsumerControl:
    """Consumer side: receives metadata, sends ACKs."""

    def __init__(self, conn: Connection):
        self._conn = conn

    def recv_metadata(self, timeout_ms: float | None = None) -> tuple[dict[str, Any], tuple] | None:
        """Receive (metadata_dict, ipc_args_tuple). Returns None on timeout."""
        timeout_s = timeout_ms / 1000.0 if timeout_ms else None
        if not self._conn.poll(timeout_s):
            return None
        try:
            msg = self._conn.recv()
            if isinstance(msg, dict) and msg.get("type") == "metadata":
                logger.debug("control: recv metadata for id=%s",
                            msg["metadata"].get("tensor_id"))
                return msg["metadata"], msg["ipc_args"]
        except EOFError:
            logger.warning("control: pipe closed")
        return None

    def send_ack(self, tensor_id: str, ack_type: str = "release") -> None:
        msg = {"type": ack_type, "tensor_id": tensor_id, "timestamp": time.monotonic()}
        self._conn.send(msg)
        logger.debug("control: sent %s for id=%s", ack_type, tensor_id)

    def close(self) -> None:
        self._conn.close()
