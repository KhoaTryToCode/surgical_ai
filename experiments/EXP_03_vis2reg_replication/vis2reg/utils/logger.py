import logging
import os
from pathlib import Path

_LOGGERS = {}


def get_logger(name: str = 'vis2reg', log_file: str | None = None):
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
