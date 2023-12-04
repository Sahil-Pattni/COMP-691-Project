"""
Creates a logger object for the project.
"""
import logging


def setup_logger(name):
    """
    Configure and return a logger with the specified name.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the logging level
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    # Create a console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter("%(name)s: %(message)s")

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    if not logger.handlers:
        logger.addHandler(ch)

    return logger
