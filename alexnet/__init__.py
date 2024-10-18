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
    # Create a file handler for logging to a file
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    if is_notebook():
        # For Colab/Jupyter, use a simple StreamHandler to stdout
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # For terminal, use RichHandler
        console_handler = RichHandler()

    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )


# Initialize logging when the package is imported
setup_logging()

__all__ = ['train', 'inference', 'test', 'utils']
