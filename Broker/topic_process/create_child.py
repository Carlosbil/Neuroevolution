import os
import json
from utils import logger, create_producer, create_consumer, produce_message
from responses import ok_message, bad_request_message, runtime_error_message, response_message, bad_model_message
from confluent_kafka import KafkaError
from database import get_population, population_exists, get_population_metadata


def process_create_child(topic, data):
    """Process messages from the 'create_child' topic and send a response to Kafka."""
    try:
        logger.info(f"Processing create_child message with data: {data}")
        
        # Validate required fields
        if 'uuid' not in data:
            logger.error("Missing required field 'uuid' in create_child data")
            bad_request_message(topic, "uuid is required")
            return None, None

        if not 'dataset_params' in data:
            logger.error("Missing required field 'dataset_params' in create_child data")
            bad_request_message(topic, "dataset_params is required")
            return None, None
        
        dataset_params = data['dataset_params']
        models_uuid = data['uuid']
        
        # Check if population exists in database
        if not population_exists(models_uuid):
            logger.error(f"Population not found in database: {models_uuid}")
            bad_model_message(topic, f"Population not found: {models_uuid}")
            return None, None

        # Get models from database
        models = get_population(models_uuid)
        
        if not models:
            logger.error(f"No models found for population: {models_uuid}")
            bad_model_message(topic, f"No models found for population: {models_uuid}")
            return None, None

        logger.info(f"Successfully loaded {len(models)} models from database for population: {models_uuid}")
        
        # Get population metadata
        logger.info(f"Getting population metadata for {models_uuid}")
        metadata = get_population_metadata(models_uuid)
        
        if not metadata:
            logger.warning(f"No metadata found for population {models_uuid}, using defaults")
            metadata = {
                'generation': 0,
                'max_generations': 10,
                'fitness_threshold': 0.95,
                'fitness_history': [],
                'best_overall_fitness': 0.0,
                'best_overall_uuid': models_uuid,
                'original_params': {}
            }
        
        json_to_send = {
            "models": models,
            "uuid": models_uuid,
            "dataset_params": dataset_params,
            "metadata": metadata
        }
        
        producer = create_producer()
        topic_to_send = "genome-create-child"
        response_topic = f"{topic_to_send}-response"
        
        logger.info(f"Sending message to {topic_to_send} topic")
        produce_message(producer, topic_to_send, json.dumps(json_to_send), times=1)
        logger.info(f"Successfully sent create child request to {topic_to_send}")
        
        return 
    except Exception as e:
        logger.error(f"Error in create_child: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Data that caused error: {data}")
        runtime_error_message(topic, f"Error in create_child: {str(e)}")
        return
