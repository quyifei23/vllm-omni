# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""pytest conftest for gpu_transport tests.

Overrides the root ``tests/conftest.py``'s ``default_vllm_config`` fixture
(which imports ``vllm`` and may fail when vLLM's transformers dependency is
mismatched).  The gpu_transport tests do not need vLLM's configuration.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def default_vllm_config():
    """Override the root conftest's default_vllm_config.

    The root fixture imports ``vllm.config`` which may fail when the
    installed ``transformers`` version is incompatible with vLLM.
    GPU transport tests do not need vLLM configuration.
    """
    return None
