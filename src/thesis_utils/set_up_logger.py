from typing import TextIO

from loguru import logger as log


def set_up_logger(stream: TextIO):
    log.remove()
    log.add(
        stream,
        colorize=True,
        format="<w>{time:DD/MM HH:mm}</w> - <level>{level}</level>: {message}\n",
    )
    return log
