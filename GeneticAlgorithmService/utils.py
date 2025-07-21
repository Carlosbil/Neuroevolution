import os
import json
import logging
import colorlog
from confluent_kafka import Producer, Consumer, KafkaError

# Load environment variables
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

# Configure logging - only use colorlog handler to avoid duplicates
logger = logging.getLogger(__name__)

# Only configure if not already configured
if not logger.handlers:
    # Configure logger with colors
    log_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

def generate_uuid():
    """Generate a unique UUID for populations."""
    import uuid
    return str(uuid.uuid4())

def create_producer():
    """Create a Kafka producer."""
    return Producer({
        'bootstrap.servers': KAFKA_BROKER,
        'linger.ms': 0,
        'batch.size': 1,
    })

def create_consumer():
    """Create and configure a Kafka consumer."""
    return Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'genetic-algorithm-consumer-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'session.timeout.ms': 60000,
        'heartbeat.interval.ms': 15000,
        'max.poll.interval.ms': 3900000,  # 65 minutes (1 hour + 5 minutes buffer)
        'fetch.min.bytes': 1,
    })

def produce_message(producer, topic, message, times=1):
    """Publish a message to Kafka."""
    for i in range(times):
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        logger.info(f"[✅] Message sent: {message} - Topic: {topic} - {i}")
