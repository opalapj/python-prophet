import logging
import logging.config
import sys
import tomllib
from pathlib import Path


_setup = Path(__file__).parent
_scripts = _setup.parent
_settings = _setup / "settings.toml"
_logging_config = _setup / "logging.toml"
_logs = _setup / "logs.log"
_data = _scripts / "data"


class Setup:
    """
    A class to recursively convert a dictionary into an object.

    :param dictionary: The dictionary to convert.
    :type dictionary: dict
    """

    def __init__(self, dictionary):
        """
        Initialize the Config object.

        :param dictionary: The dictionary to convert.
        :type dictionary: dict
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Setup(value)
            self.__setattr__(key, value)


def _configure_logging():
    """
    Logging configuration from a TOML file.

    Opens the logging configuration file and applies the logging settings.
    If the file handler's filename is not absolute, it sets the filename
    to be relative to config directory.
    """
    with _logging_config.open("rb") as file:
        content = tomllib.load(file)
    filename = Path(content["handlers"]["exceptions"]["filename"])
    if not filename.is_absolute():
        content["handlers"]["exceptions"]["filename"] = _setup / filename
    logging.config.dictConfig(config=content)


def _log_uncaught_exception():
    """
    Set up logging for uncaught exceptions.

    Wraps the default sys.excepthook with a decorator that logs uncaught
    exceptions.
    """

    def with_logging(excepthook):
        _catcher = logging.getLogger("catcher")

        def wrapper(*exc_info):
            _catcher.error("Uncaught exception:", exc_info=exc_info)
            excepthook(*exc_info)

        return wrapper

    sys.excepthook = with_logging(sys.__excepthook__)


def _load_settings():
    """
    Load user settings from a TOML file.

    Reads the settings file and loads its contents into a dictionary.

    :return: The settings from the TOML file.
    :rtype: dict
    """
    with _settings.open("rb") as file:
        content = tomllib.load(file)
    return content


def _collect_pathlike_variables():
    """
    Collect path-like variables from the global namespace.

    Scans global variables and collects those that are instances of Path.

    :return: A dictionary of path-like variables.
    :rtype: dict
    """
    variables = {}
    for name, value in globals().items():
        if isinstance(value, Path):
            variables[name.removeprefix("_")] = value
    return variables


_configure_logging()
_log_uncaught_exception()
settings = Setup(_load_settings())
locations = Setup(_collect_pathlike_variables())
