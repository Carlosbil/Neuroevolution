import os
import json
import requests
from utils import logger, generate_uuid, check_initial_poblation, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message
from confluent_kafka import KafkaError

def process_create_initial_population(topic, data):
    """Process messages from the 'create_initial_population' topic and send a response to Kafka."""
    try:
        logger.info(f"Processing create_initial_population with data: {data}")
        
        # Validate input data
        if not check_initial_poblation(data):
            missing_keys = []
            required_keys = ['num_channels', 'px_h', 'px_w', 'num_classes', 'batch_size', 'num_poblation']
            for key in required_keys:
                if key not in data:
                    missing_keys.append(key)
            
            error_msg = f"Invalid initial population parameters. Missing keys: {missing_keys}" if missing_keys else "Invalid initial population parameters"
            logger.error(error_msg)
            logger.error(f"Received data: {data}")
            bad_model_message(topic, error_msg)
            return None, None

        # Send the message to the genome-create-initial-population topic
        json_to_send = data
        producer = create_producer()
        topic_to_sed = "genome-create-initial-population"
        response_topic = f"{topic_to_sed}-response"
        
        logger.info(f"Sending message to {topic_to_sed} topic")
        produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
        logger.info(f"Successfully sent initial population request to {topic_to_sed}")
        return None, None
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Data that caused error: {data}")
        return runtime_error_message(topic, f"Error in create_initial_population: {str(e)}")