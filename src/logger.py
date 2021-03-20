import logging
from logging import Logger

from settings import *


def get_logger(name: str=None) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    sh = logging.StreamHandler()
    sh.setLevel(INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger


logger = get_logger()