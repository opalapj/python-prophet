"""
Main script to test configuration loading and functionality of the
developing package.

This script executes several functions to test logging, package
functionality, and configuration loading. It imports modules from
the package and prints configuration settings and locations.
"""

import logging

from forecast import my_module

from setup import locations
from setup import settings


logger = logging.getLogger(__name__)


def foo():
    """
    Log a test debug message.

    This function logs a debug message to verify that the logging setup
    is working correctly for the main script.
    """
    logger.debug("Test debug from main script.")
    logger.warning("Test warning from main script.")


def main():
    """
    Main function to execute test operations.

    This function executes foo(), package.bar(), module.baz(), prints
    the settings and locations, and intentionally raises a ZeroDivisionError
    to test error handling.
    """
    foo()
    my_module.bar()
    print(f"Hi {settings.first_name} {settings.last_name}!")
    print(f"{settings.__dict__ = }")  # noqa: E202, E251
    print(f"{locations.__dict__ = }")  # noqa: E202, E251
    print((locations.data / "lines").read_text(encoding="utf-8"))
    print("Erroneous statement:", 1 / 0)


if __name__ == "__main__":
    main()
