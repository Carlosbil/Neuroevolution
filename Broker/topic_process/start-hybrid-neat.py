from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError
from utils import get_storage_path


def process_start_hybrid_neat(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing start_hybrid_neat")
        
        data = {
            "num_channels": 1,
            "px_h": 28,
            "px_w": 28,
            "num_classes": 10,
            "batch_size": 32,
            "num_poblation": 10
        }
        
        json_to_send = data
        producer = create_producer()
        topic_to_sed = "create-initial-population"
        produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
        
        return
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        runtime_error_message(topic)
        return None, None