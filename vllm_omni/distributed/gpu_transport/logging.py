# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os

_DEBUG = os.environ.get("GPU_CONNECTOR_DEBUG", "0") == "1"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if _DEBUG:
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(h)
    return logger
