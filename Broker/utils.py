import logging
import colorlog
import uuid
import os
import json
import argparse
from confluent_kafka import Producer, Consumer, KafkaException, KafkaError
import signal
import sys
import time

# Environment variable configuration
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
STORAGE_PATH = os.environ.get("BROKER_STORAGE_PATH", os.path.join(os.path.dirname(__file__), 'models'))

def parse_args():
    """
    Parse command-line arguments for the broker service.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Broker service for Neuroevolution')
    parser.add_argument('--storage-path', type=str, help='Path to store model files')
    return parser.parse_args()

def get_storage_path():
    """
    Get the configured storage path for saving model files.
    Ensures the directory exists.

    :return: Absolute path to the directory used for storing model files.
    :rtype: str
    """
    args = parse_args()
    if args.storage_path:
        path = args.storage_path
    else:
        path = STORAGE_PATH
    
    os.makedirs(path, exist_ok=True)
    return path

# Global logger configuration with file name and line number
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

# Create a color log handler for console output
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
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
    Check if the initial population parameters are valid.

    :param params: Dictionary containing population configuration parameters.
    :type params: dict
    :return: True if all required parameters are present, False otherwise.
    :rtype: bool
    """
    required_keys = ['num_channels', 'px_h', 'px_w', 'num_classes', 'batch_size', 'num_poblation']
    return all(key in params for key in required_keys)

def generate_uuid():
    """
    Generate a random UUID.

    :return: A string representing the UUID.
    :rtype: str
    """
    return str(uuid.uuid4())

def get_possible_models(models_uuid):
    """
    Retrieve the list of possible models from a JSON file using the provided UUID.

    :param models_uuid: UUID of the model set.
    :type models_uuid: str
    :return: Dictionary of possible models loaded from file.
    :rtype: dict
    """
    path = os.path.join(get_storage_path(), f'{models_uuid}.json')
    with open(path, 'r') as file:
        possible_models = json.load(file)
    return possible_models

def produce_message(producer, topic, message, times=10):
    """
    Publish a message to a Kafka topic multiple times.

    :param producer: Kafka producer instance.
    :type producer: confluent_kafka.Producer
    :param topic: Kafka topic name.
    :type topic: str
    :param message: Message to be published.
    :type message: str
    :param times: Number of times the message should be sent.
    :type times: int
    """
    for i in range(times):
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        logger.info(f"✅ Mensaje enviado: {message} - Tópico: {topic} - {i} ")

def create_producer():
    """
    Create and configure a Kafka producer.

    :return: A configured Kafka producer instance.
    :rtype: confluent_kafka.Producer
    """
    return Producer({
        'bootstrap.servers': KAFKA_BROKER,
        'linger.ms': 0,
        'batch.size': 1,
    })

def create_consumer():
    """
    Create and configure a Kafka consumer with optimized error handling.

    :return: A configured Kafka consumer instance.
    :rtype: confluent_kafka.Consumer
    """
    return Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'broker-consumer-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'session.timeout.ms': 60000,
        'heartbeat.interval.ms': 15000,
        'max.poll.interval.ms': 300000,
    })
