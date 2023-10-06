import datetime
from enum import Enum


class Ccodes(Enum):
    RESET = "\033[0m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"


def log(message, color=Ccodes.RESET):
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S.%f]")
    print(f"{timestamp} {color.value}{message}{Ccodes.RESET.value}")
