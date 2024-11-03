def cc(color, text):
    colors = {
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "FUCHSIA": "\033[95m",
        "GRAY": "\033[90m",
        "ENDC": "\033[0m",
    }

    return f"{colors[color]}{text}{colors['ENDC']}"
