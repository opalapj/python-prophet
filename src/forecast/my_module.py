"""
This module provides functionality for ...
"""

import logging


logger = logging.getLogger(__name__)


def bar():
    """
    Log a test debug message.

    This function logs a debug message to verify that the logging setup
    is working correctly for the imported module.
    """
    logger.debug("Test debug from imported module.")
    logger.warning("Test warning from imported module.")
