import logging
import colorlog
import uuid
import os
import json

# Configure global logger only once.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Avoid duplicate log messages: clear any existing handlers and disable propagation.
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

# Create a colored log handler for console output.
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

def check_initial_poblation(params):
    """ 
    Check the initial poblation parameters.
    """
    if 'num_channels' not in params or 'px_h' not in params or 'px_w' not in params or 'num_classes' not in params or 'batch_size' not in params or 'num_poblation' not in params:
        return False
    return True

def generate_uuid():
    """ 
    Generate a random UUID.
    """
    return str(uuid.uuid4())

def get_possible_models(models_uuid):
    """ 
    Get the possible models from the given UUID.
    """
    # Load possible models from ./models/uuid.json
    path = os.path.join(os.path.dirname(__file__), 'models', f'{models_uuid}.json')
    with open(path, 'r') as file:
        possible_models = json.load(file)
    return possible_models