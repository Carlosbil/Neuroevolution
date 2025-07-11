from utils import logger, create_producer, produce_message, create_consumer, generate_uuid
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
from database import get_population, population_exists, save_population, save_model
import json
import random
import torch
from topic_process.evaluate_population import process_evaluate_population


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

# Maps from Genome module
MAP_ACTIVATE_FUNCTIONS = {
    'relu': 'relu',
    'sigmoid':'sigmoid',
    'tanh': 'tanh',
    'leakyrelu': 'leakyrelu',
}

MAP_OPTIMIZERS = {
    'adam': 'adam',
    'adamw': 'adamw',
    'sgd': 'sgd',
    'rmsprop': 'rmsprop',
}

def cross_genomes_local(genome1, genome2):
    """
    Function to cross two genomes. 50% of the time, the gene will be taken from genome1, otherwise from genome2. Randomly.
    This matches exactly the implementation in Genome > utils.py
    """
    logger.debug(f"Crossing genomes: {genome1} and {genome2}")
    new_genome = {}
    for key in genome1:
        new_genome[key] = genome1[key] if torch.rand(1) > 0.5 else genome2[key]
        
    return new_genome

def mutate_genome_local(genome, mutation_rate=0.1):
    """
    Function to mutate a genome. Randomly changes a gene with a given mutation rate.
    This matches exactly the implementation in Genome > utils.py
    """
    for key in genome:
        if torch.rand(1) < mutation_rate:
            logger.debug(f"Mutating key: {key} with value: {genome[key]}")
            
            if key in ['Number of Convolutional Layers', 'Number of Fully Connected Layers']:
                genome[key] = torch.randint(1, 5, (1,)).item()
            
            elif key == 'Number of Nodes in Each Layer':
                genome[key] = [torch.randint(16, 128, (1,)).item() for _ in range(genome['Number of Fully Connected Layers'])]
            
            elif key == 'Activation Functions':
                # Selección aleatoria de funciones de activación
                activation_function_key = random.choice(list(MAP_ACTIVATE_FUNCTIONS.keys()))  # Selección aleatoria de la clave
                genome[key] = [activation_function_key for _ in range(genome['Number of Fully Connected Layers'])]  # Asigna las funciones de activación
            
            elif key in ['Dropout Rate', 'Learning Rate']:
                factor = random.choice([0.1, 0.01, 0.001])
                genome[key] *= factor if random.random() < 0.5 else 1 / factor
                genome[key] = min(round(genome[key], 4), 0.9)  # Asegurar que no supere 1
            
            elif key == 'Batch Size':
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'Optimizer':
                # Selección aleatoria de un optimizador del mapa
                optimizer_key = random.choice(list(MAP_OPTIMIZERS.keys()))  # Selección aleatoria de la clave
                genome[key] = optimizer_key  # Asigna el nombre del optimizador
            
            elif key in ['num_channels', 'num_classes']:
                genome[key] = torch.randint(1, 4, (1,)).item() if key == 'num_channels' else torch.randint(2, 10, (1,)).item()
            
            elif key in ['px_h', 'px_w']:
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'filters':
                genome[key] = [torch.randint(1, 5, (1,)).item() for _ in range(genome['Number of Convolutional Layers'])]
            
            elif key == 'kernel_sizes':
                genome[key] = [torch.randint(3, 7, (1,)).item() for _ in range(genome['Number of Convolutional Layers'])]
            
            logger.debug(f"New value for {key}: {genome[key]}")
    
    return genome

def process_select_best_architectures(topic, data):
    """Process messages to select the best 50% of architectures using tournament selection.
    Args:
        topic (str): The Kafka topic
        data (dict): Message data containing UUID of the population
    Returns:
        tuple: (models_uuid, new_uuid) or (None, None) on error
    """
    try:
        logger.info("Processing select_best_architectures")
        if 'uuid' not in data:
            bad_request_message(topic, "uuid is required")
            return None, None
        
        # Extract the models
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
        
        # Ensure all models have a score
        for model in models.values():
            if 'score' not in model:
                model['score'] = 0

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

        # Save selected models to database with new UUID
        new_uuid = f"{models_uuid}_best50percent"
        save_population(new_uuid)
        
        # Save each selected model to the new population
        for model_id, model_data in selected_models.items():
            save_model(new_uuid, model_id, model_data)

        logger.info(f"Selected {len(selected_models)} best models and saved to new population: {new_uuid}")
        
        message = {
            "uuid": new_uuid,
            "message": f"Best {len(selected_models)} architectures selected successfully",
            "selected_models": list(selected_models.keys()),
            "original_population_size": population_size,
            "selected_population_size": len(selected_models)
        }
        
        ok_message(topic, message)
        return new_uuid, new_uuid

    except Exception as e:
        logger.error(f"Error in select_best_architectures: {e}")
        runtime_error_message(topic)
        return None, None