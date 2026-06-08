"""
NVTX instrumentation utilities for vLLM-Omni Nsight Systems profiling.

Uses the same env var as upstream vLLM (VLLM_NVTX_SCOPES_FOR_PROFILING) so
that a single flag enables both vllm and vllm-omni NVTX markers in nsys.

Two annotation modes:
- ``nvtx_range(name)`` — push/pop range (duration measurement), use for
  operations that take measurable time.
- ``nvtx_mark(name)`` — single timestamp marker, use for lightweight
  events or loop-iteration markers where duration is not interesting.

Usage:
    from vllm_omni.utils.nvtx import nvtx_range, nvtx_mark

    with nvtx_range("omni:heavy_op"):
        result = heavy_op()

    nvtx_mark("omni:checkpoint")

Enable with: VLLM_NVTX_SCOPES_FOR_PROFILING=1
"""

import contextlib
import os
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

_RANGE_FUNC: Callable[..., Any] | None = None
_MARK_FUNC: Callable[..., Any] | None = None


def _resolve_funcs() -> None:
    """Resolve annotation / mark functions once (lazy, on first call)."""
    global _RANGE_FUNC, _MARK_FUNC
    if _RANGE_FUNC is not None:
        return

    if bool(int(os.environ.get("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))):
        try:
            import nvtx
        except ModuleNotFoundError:
            logger.warning(
                "VLLM_NVTX_SCOPES_FOR_PROFILING=1 but the 'nvtx' package is "
                "not installed. Install it with: pip install nvtx. "
                "NVTX profiling will be disabled for this session."
            )
            _RANGE_FUNC = lambda *a, **kw: contextlib.nullcontext()
            _MARK_FUNC = lambda *a, **kw: None
            return

        _RANGE_FUNC = nvtx.annotate
        _MARK_FUNC = nvtx.mark
        logger.info(
            "vLLM-Omni NVTX profiling ENABLED (VLLM_NVTX_SCOPES_FOR_PROFILING=1). "
            "Omni ranges & marks will appear in Nsight Systems."
        )
    else:
        _RANGE_FUNC = lambda *a, **kw: contextlib.nullcontext()
        _MARK_FUNC = lambda *a, **kw: None
        logger.info(
            "vLLM-Omni NVTX profiling DISABLED. "
            "Set VLLM_NVTX_SCOPES_FOR_PROFILING=1 to enable."
        )


def nvtx_range(name: str) -> contextlib.AbstractContextManager:
    """Push/pop an NVTX range. Returns a context manager that marks duration.

    Use for operations that take measurable time.  No-op when profiling
    is disabled.
    """
    _resolve_funcs()
    return _RANGE_FUNC(name)


def nvtx_mark(name: str) -> None:
    """Emit a single NVTX timestamp marker (zero-duration event).

    Use for lightweight checkpoints / loop-iteration markers.
    No-op when profiling is disabled.
    """
    _resolve_funcs()
    _MARK_FUNC(name)
