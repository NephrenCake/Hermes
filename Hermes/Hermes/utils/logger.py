import logging
import sys

from logging import Logger

from Hermes.platform.env import LOG_LEVEL

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        # Calculate the fixed width for the prefix (levelname + asctime + [filename:lineno])
        # This is an estimate - you may need to adjust based on your actual maximum lengths
        self.prefix_width = 50  # Adjust this value as needed

    def format(self, record):
        # First format the record normally
        msg = super().format(record)

        # Calculate the actual prefix length up to the message
        prefix_end = msg.find(record.message)
        if prefix_end == -1:
            return msg

        # Get the prefix part (everything before the message)
        prefix = msg[:prefix_end]

        # If the prefix is shorter than our desired width, pad it with spaces
        if len(prefix) < self.prefix_width:
            msg = prefix.ljust(self.prefix_width) + record.message
        else:
            msg = prefix + record.message

        # Handle newlines in the message
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + " " * self.prefix_width)

        return msg


_root_logger = logging.getLogger("Hermes")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.DEBUG)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str) -> Logger:
    logger = logging.getLogger(name)

    logger.setLevel(eval(f"logging.{LOG_LEVEL}"))
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger
