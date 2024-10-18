from alexnet.model import AlexNet
from alexnet.test import test
from alexnet.train import train
from alexnet.inference import inference
import alexnet.utils as utils
import logging
from rich.logging import RichHandler
import sys

# Check if the code is running in a Jupyter notebook
def is_notebook():
    try:
        # Check if IPython is installed and if we are in a Jupyter environment
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# Set up the logging configuration
def setup_logging():

    # Get the root logger
    logger = logging.getLogger()

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    
    # Create a formatter for the file handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Use RichHandler for the console output, but detect if we're in Jupyter
    if is_notebook():
        # Use RichHandler and rich tracebacks for Jupyter
        console_handler = RichHandler(rich_tracebacks=True)
    else:
        # Use RichHandler for terminals
        console_handler = RichHandler()

    # Set the logging format for console output (simplified for clarity)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,  # Set logging level
        handlers=[console_handler, file_handler]  # Log to both console and file
    )

# Initialize logging when the package is imported
setup_logging()

__all__ = ['train', 'inference', 'test', 'utils']
