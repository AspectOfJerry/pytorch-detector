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

    # Automatically color boolean values
    text = text.replace("True", f"{colors['GREEN']}True{colors['ENDC']}")
    text = text.replace("False", f"{colors['RED']}False{colors['ENDC']}")

    return f"{colors[color]}{text}{colors['ENDC']}"


def ccnum(number, reverse=False):
    green = "\033[92m"
    red = "\033[91m"
    endc = "\033[0m"

    if reverse:
        # Color the number red if it is positive, green if it is negative
        return f"{green}{number}{endc}" if number < 0 else f"{red}{number}{endc}"
    else:
        # Color the number green if it is positive, red if it is negative
        return f"{red}{number}{endc}" if number < 0 else f"{green}{number}{endc}"
