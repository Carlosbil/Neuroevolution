from utils import logger, create_producer, produce_message, create_consumer, generate_uuid
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message, response_message
import requests
import json
import os
import random
from confluent_kafka import KafkaError
from topic_process.create_initial_population import process_create_initial_population
from topic_process.evaluate_population import process_evaluate_population
from topic_process.select_best_architectures import process_select_best_architectures
from topic_process.create_child import process_create_child

TOPIC_PROCESSORS = {
    "create-child": "create-child",
    "create-initial-population": "create-initial-population",
    "evaluate-population": "evaluate-population",
    "genetic-algorithm": "genetic-algorithm",
    "select-best-architectures": "select-best-architectures",
}


def create_new_children(topic, models_uuid, num_children=5):
    """Create new children by combining models from the selected population.
    
    Args:
        topic (str): The Kafka topic
        models_uuid (str): UUID of the selected models
        num_children (int): Number of children to create
        
    Returns:
        tuple: (new_models_uuid, new_path) or (None, None) on error
    """
    try:
        logger.info(f"Creating {num_children} new children from selected models")
        
        # Load the selected models
        path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            logger.error(f"Models file not found: {path}")
            return None, None
            
        with open(path, 'r') as file:
            selected_models = json.load(file)
            
        # Create a new dictionary to store the children
        new_children = {}
        
        # Get the list of model IDs
        model_ids = list(selected_models.keys())
        if len(model_ids) < 2:
            logger.error("Need at least 2 models to create children")
            return None, None
            
        # Create the specified number of children
        for i in range(num_children):
            # Randomly select two parent models
            parent1_id = random.choice(model_ids)
            parent2_id = random.choice([m_id for m_id in model_ids if m_id != parent1_id])
            
            # Call process_create_child to create a new child
            child_data = {
                "model_id": parent1_id,
                "second_model_id": parent2_id,
                "uuid": models_uuid
            }
            
            child_result = process_create_child(TOPIC_PROCESSORS["create-child"], child_data)
            if child_result is None:
                logger.error(f"Failed to create child {i+1}")
                continue
                
            # Add the child to our new children dictionary
            child_id = f"child_{i+1}"
            new_children[child_id] = child_result.get("message", {})
        
        if not new_children:
            logger.error("Failed to create any children")
            return None, None
            
        # Save the new children to a file
        new_uuid = f"{models_uuid}_children"
        new_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{new_uuid}.json')
        
        with open(new_path, 'w') as file:
            json.dump(new_children, file, indent=4)
            
        logger.info(f"Created {len(new_children)} new children, saved to {new_path}")
        return new_uuid, new_path
        
    except Exception as e:
        logger.error(f"Error in create_new_children: {e}")
        return None, None


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
        
        # Create 5 new children from the selected models
        children_uuid, children_path = create_new_children(topic, models_uuid, num_children=5)
        if children_uuid is None:
            return response_message(topic, "Error creating child models", 500)
        
        return ok_message(topic, {
            "uuid": models_uuid, 
            "path": path, 
            "children_uuid": children_uuid,
            "children_path": children_path,
            "message": "Genetic algorithm completed successfully: population created, evaluated, best 50% selected, and 5 new children created"
        })
    except Exception as e:
        logger.error(f"Error in genetic_algorithm: {e}")
        return runtime_error_message(topic)
    
import time