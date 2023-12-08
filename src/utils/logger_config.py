"""
Creates a logger object for the project.
"""
import logging


class LoggerCustom:
    """
    A singleton class to create a logger object for the project.
    """

    # The logger object
    logger = None

    @staticmethod
    def get_logger(level=logging.INFO):
        """
        Return the logger object.
        """
        if LoggerCustom.logger is None:
            LoggerCustom.logger = LoggerCustom.__setup_logger("Logger", level)
        else:
            LoggerCustom.logger.setLevel(level)
        return LoggerCustom.logger

    @staticmethod
    def set_level(level):
        """
        Set the logging level.
        """
        LoggerCustom.logger.setLevel(level)

    # Singleton class
    def __new__(cls):
        raise NotImplementedError("This class cannot be instantiated.")

    def __setup_logger(name, level: int = logging.INFO):
        """
        Configure and return a logger with the specified name.
        """
        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(level)  # Set the logging level
        # Prevent the logger from propagating messages to the root logger
        logger.propagate = False

        # Create a console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter ([level] name: <message>])
        formatter = logging.Formatter("[%(levelname)s] %(name)s: <%(message)s>")

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        if not logger.handlers:
            logger.addHandler(ch)

        return logger


# Runs at import time
logger = LoggerCustom.get_logger(level=logging.INFO)
