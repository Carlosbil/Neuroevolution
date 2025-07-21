import os
import json
import requests
from database import save_population_with_metadata, save_population
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

        # Generate UUID for this population before sending to Genome
        models_uuid = generate_uuid()
        logger.info(f"Generated UUID for initial population: {models_uuid}")
        
        # Send the message to the genome-create-initial-population topic
        json_to_send = data.copy()  # Make a copy to avoid modifying original
        json_to_send['population_uuid'] = models_uuid  # Include the UUID in the message
        
        producer = create_producer()
        topic_to_sed = "genome-create-initial-population"
        response_topic = f"{topic_to_sed}-response"
        
        save_population(models_uuid)

        save_population_with_metadata(
            population_uuid=models_uuid,
            generation=0,
            max_generations=data['max_generations'],
            fitness_threshold=data['fitness_threshold'],
            fitness_history=[],
            best_overall_fitness=0,
            best_overall_uuid=models_uuid,
            original_params=data  # Store all original parameters including path
        )
        
        
        logger.info(f"Sending message to {topic_to_sed} topic with UUID: {models_uuid}")
        produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
        logger.info(f"Successfully sent initial population request to {topic_to_sed}")
        
        # Return the UUID immediately so genetic_algorithm can use it
        return models_uuid, None
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Data that caused error: {data}")
        return runtime_error_message(topic, f"Error in create_initial_population: {str(e)}")