import os
import json
import requests
from utils import logger, generate_uuid, check_initial_poblation, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message
from confluent_kafka import KafkaError

def process_create_initial_population(topic, data):
    """Process messages from the 'create_initial_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing create_initial_population")
        if not check_initial_poblation(data):
            bad_model_message(topic)
            return None, None

        # Send the message to the genome-create-initial-population topic
        json_to_send = data
        producer = create_producer()
        topic_to_sed = "genome-create-initial-population"
        response_topic = f"{topic_to_sed}-response"
        produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
        # get the models
        return
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        return runtime_error_message(topic)