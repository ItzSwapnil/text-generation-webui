import logging
import platform
import ctypes
import colorama  # Windows console coloring
import termcolor  # Non-Windows console coloring

# Define constants for the different colors
COLOR_BLACK = 0x0000
COLOR_BLUE = 0x0001
COLOR_GREEN = 0x0002
COLOR_CYAN = 0x0003
COLOR_RED = 0x0004
COLOR_MAGENTA = 0x0005
COLOR_YELLOW = 0x0006
COLOR_WHITE = 0x0007
COLOR_INTENSITY = 0x0008

# Define a constant to reset the color to the default
RESET_COLOR = COLOR_WHITE

# Define a class to handle colored logging on Windows
class ColorHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.out_handle = ctypes.windll.kernel32.GetStdHandle(-11)

    def emit(self, record):
        levelno = record.levelno
        color = None

        if levelno >= 50:
            color = COLOR_RED | COLOR_INTENSITY
        elif levelno >= 40:
            color = COLOR_RED | COLOR_INTENSITY
        elif levelno >= 30:
            color = COLOR_YELLOW | COLOR_INTENSITY
        elif levelno >= 20:
            color = COLOR_GREEN
        elif levelno >= 10:
            color = COLOR_MAGENTA
        else:
            color = COLOR_WHITE

        if color is not None:
            if not ctypes.windll.kernel32.SetConsoleTextAttribute(self.out_handle, color):
                # If the function fails, print an error message and reset the color to the default
                print(f"Error setting console text attribute to {color}: {ctypes.FormatError()}")
                color = RESET_COLOR

        super().emit(record)

        if color is not None:
            # Reset the color to the default
            ctypes.windll.kernel32.SetConsoleTextAttribute(self.out_handle, RESET_COLOR)

# Define a function to set up the logger
def setup_logging():
    # Create a logger
    logger = logging.getLogger('text-generation-webui')
    logger.setLevel(logging.DEBUG)

    # Create a console handler with coloring
    if platform.system() == 'Windows':
        # Use the ColorHandler class on Windows
        handler = ColorHandler()
    else:
        # Use the termcolor library on non-Windows platforms
        handler = logging.StreamHandler()
        handler.setFormatter(termcolor.colorize_formatter(logger.handlers[0].formatter))

    # Add the handler to the logger
    logger.addHandler(handler)

# Call the setup_logging function to set up the logger
setup_logging()

# Test the logger
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
