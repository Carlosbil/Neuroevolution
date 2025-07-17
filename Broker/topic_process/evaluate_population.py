from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError
from database import get_population, population_exists


def process_evaluate_population(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    try:
        logger.info(f"Processing evaluate_population with data: {data}")
        
        # Validate input data
        if 'uuid' not in data:
            logger.error("Missing required field 'uuid' in evaluate_population data")
            bad_request_message(topic, "uuid is required")
            return None, None
        
        # Extract the population UUID
        models_uuid = data['uuid']
        logger.info(f"Looking for population with UUID: {models_uuid}")
        
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
        logger.debug(f"Model keys: {list(models.keys())}")

        # For each model, send a message to the evolutioner-create-cnn-model topic
        models_sent = 0
        for model_id, model_data in models.items():
            try:
                # Ensure model_data is a dictionary
                if not isinstance(model_data, dict):
                    logger.error(f"Model data for {model_id} is not a dictionary: {type(model_data)}")
                    continue
                
                # extract the model
                json_to_send = model_data.copy()  # Make a copy to avoid modifying original
                json_to_send["model_id"] = model_id
                json_to_send["uuid"] = models_uuid
                
                logger.debug(f"Sending model {model_id} to evolutioner-create-cnn-model")
                logger.debug(f"Model data keys: {list(model_data.keys())}")
                
                producer = create_producer()
                topic_to_sed = "evolutioner-create-cnn-model"
                response_topic = f"{topic_to_sed}-response"
                produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
                models_sent += 1
                
            except Exception as model_error:
                logger.error(f"Error processing model {model_id}: {model_error}")
                logger.error(f"Model data that caused error: {model_data}")
                logger.error(f"Model data type: {type(model_data)}")
                continue
        
        logger.info(f"Successfully sent {models_sent} models to evolutioner-create-cnn-model topic")
        
        # Return the UUID immediately after sending models for evaluation
        # The evaluation results will be processed asynchronously via responses
        return models_uuid, None
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Data that caused error: {data}")
        runtime_error_message(topic, f"Error in evaluate_population: {str(e)}")
        return None, None