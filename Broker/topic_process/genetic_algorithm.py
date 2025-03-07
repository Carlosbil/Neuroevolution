from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message, response_message
import requests
import json
import os
from confluent_kafka import KafkaError
from topic_process.create_initial_population import process_create_initial_population
from topic_process.evaluate_population import process_evaluate_population
from topic_process.select_best_architectures import process_select_best_architectures
TOPIC_PROCESSORS = {
    "create-child": "create-child",
    "create-initial-population": "create-initial-population",
    "evaluate-population": "evaluate-population",
    "genetic-algorithm": "genetic-algorithm",
    "select-best-architectures": "select-best-architectures",
}


def process_genetic_algorithm(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing evaluate_population")
        
        # First create an initial population

        models_uuid, path = process_create_initial_population(TOPIC_PROCESSORS["create-initial-population"], data)
        if models_uuid is None:
            return response_message(topic, "Error creating initial population", 500)
        
        json_to_send = {"uuid": models_uuid}
        # Now we evaluate the population
        models_uuid, path = process_evaluate_population(TOPIC_PROCESSORS["evaluate-population"], json_to_send)
        if models_uuid is None:
            return response_message(topic, "Error evaluating population", 500)
        
        # Select the best 50% of the population
        json_to_send = {"uuid": models_uuid}
        models_uuid, path = process_select_best_architectures(TOPIC_PROCESSORS["select-best-architectures"], json_to_send)
        if models_uuid is None:
            return response_message(topic, "Error selecting best architectures", 500)
        
        return ok_message(topic, {"uuid": models_uuid, "path": path, "message": "Genetic algorithm completed successfully: population created, evaluated, and best 50% selected"})
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        return runtime_error_message(topic)
    
import time