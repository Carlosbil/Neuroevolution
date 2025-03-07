from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import json
import os
import random

def tournament_selection(population, k=3):
    """Perform tournament selection on the population.
    Args:
        population (dict): Dictionary of models with their scores
        k (int): Tournament size
    Returns:
        str: ID of the winner model
    """
    tournament = random.sample(list(population.items()), k)
    winner = max(tournament, key=lambda x: float(x[1].get('score', 0)) if isinstance(x[1].get('score', 0), (int, float)) else 0)
    return winner[0]

def process_select_best_architectures(topic, data):
    """Process messages to select the best 50% of architectures using tournament selection.
    Args:
        topic (str): The Kafka topic
        data (dict): Message data containing UUID of the population
    Returns:
        tuple: (models_uuid, path) or (None, None) on error
    """
    try:
        logger.info("Processing select_best_architectures")
        if 'uuid' not in data:
            bad_request_message(topic, "uuid is required")
            return None, None
        
        # Extract the models
        models_uuid = data['uuid']
        path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            bad_model_message(topic)
            return None, None

        with open(path, 'r') as file:
            models = json.load(file)
        
        # Validate that models have scores
        if not all('score' in model for model in models.values()):
            bad_request_message(topic, "All models must have scores")
            return None, None

        # Select best 50% of architectures using tournament selection
        selected_models = {}
        population_size = len(models)
        tournament_size = min(3, population_size)  # Adjust tournament size for small populations
        
        # Calculate number of models to select (50% of population)
        num_to_select = max(1, population_size // 2)  # At least select 1 model
        
        for _ in range(num_to_select):  # Select 50% of the population
            if not models:  # If all models have been selected
                break
            winner_id = tournament_selection(models, tournament_size)
            selected_models[winner_id] = models[winner_id]
            models.pop(winner_id)  # Remove winner from candidate pool

        # Save selected models to a new file
        new_uuid = f"{models_uuid}_best50percent"
        new_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{new_uuid}.json')
        
        with open(new_path, 'w') as file:
            json.dump(selected_models, file, indent=4)

        message = {
            "uuid": new_uuid,
            "path": new_path,
            "message": f"Best {len(selected_models)} architectures (50% of population) selected successfully",
            "selected_models": list(selected_models.keys())
        }
        ok_message(topic, message)
        return new_uuid, new_path

    except Exception as e:
        logger.error(f"Error in select_best_architectures: {e}")
        runtime_error_message(topic)
        return None, None