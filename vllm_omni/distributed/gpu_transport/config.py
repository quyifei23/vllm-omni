# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal

TransportMode = Literal["cuda_ipc", "cuda_copy", "none"]


@dataclass
class GPUTransportConfig:
    mode: TransportMode = "cuda_ipc"
    src_device: int = 0
    dst_device: int = 1
    release_timeout_ms: float = 10_000.0
    enable_peer_access: bool = True
    ack_conn: Connection | None = None  # for ACK thread (producer side)
    consumer_ack_conn: Connection | None = None  # for sending ACKs (consumer side)
