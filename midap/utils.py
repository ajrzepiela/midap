import logging
import sys
import os

def get_logger(filepath, logging_level=7):
    """
    Get logger, from given file path and with logging level
    :param filepath: name of the file that is calling the logger, used to give it a name.
    :param logging_level: A number from 0 to 7 indicating the amount of output, defaults to 7 (debug output)
    :return: logger object
    """

    logger = logging.getLogger(os.path.basename(filepath))

    if len(logger.handlers) == 0:
        log_formatter = logging.Formatter(fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ",
                                          datefmt="%y-%m-%d %H:%M:%S", style='%')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
        set_logger_level(logger, logging_level)

    return logger


def set_logger_level(logger, level):
    """
    Sets the level of a logger
    :param logger: The logger object to set the level
    :param level: The level, a number from 0 to 7 (corresponding to the log level in the bash script)
    """

    if level <= 2:
        log_level = logging.CRITICAL
    elif level == 3:
        log_level = logging.ERROR
    elif level == 4:
        log_level = logging.WARNING
    elif level <= 6:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)


def get_inheritors(klass):
    """
    Get all child classes of a given class
    :param klass: The class to get all children
    :return: All children as a set
    """

    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses