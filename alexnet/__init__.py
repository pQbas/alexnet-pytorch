from alexnet.model import AlexNet
from alexnet.test import test
from alexnet.train import train
from alexnet.inference import inference
import alexnet.utils as utils
import logging
import colorlog
from rich.logging import RichHandler


# Set up the logging configuration
def setup_logging():
    # Configure the root logger to use RichHandler
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(message)s",  # Simplify the log format
        datefmt="[%X]",  # Set date format
        handlers=[RichHandler()]  # Use RichHandler for colored output
    )
# def setup_logging():
#     # Create a color formatter
#     formatter = colorlog.ColoredFormatter(
#         "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         log_colors={
#             'DEBUG':    'cyan',
#             'INFO':     'green',
#             'WARNING':  'yellow',
#             'ERROR':    'red',
#             'CRITICAL': 'bold_red',
#         }
#     )
#
#     # Set up the handler for logging to console
#     handler = logging.StreamHandler()
#     handler.setFormatter(formatter)
#
#     # Set up the logger
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)  # Set to the lowest level you want to log
#     logger.addHandler(handler)
#
# Initialize logging when the package is imported
setup_logging()

__all__ = ['train', 'inference', 'test', 'utils']
