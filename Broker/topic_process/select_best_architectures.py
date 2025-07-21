from utils import logger, create_producer, produce_message, create_consumer, generate_uuid
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
from database import get_population, population_exists, save_population, save_model, get_population_metadata, save_population_with_metadata
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
            
            elif key == 'score':
                genome[key] = 0.0

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

        # Get the best fitness from the current population
        current_best_fitness = 0.0
        current_best_model_id = None
        for model_id, model_data in models.items():
            model_score = float(model_data.get('score', 0)) if isinstance(model_data.get('score', 0), (int, float)) else 0.0
            if model_score > current_best_fitness:
                current_best_fitness = model_score
                current_best_model_id = model_id

        logger.info(f"Current population best fitness: {current_best_fitness:.4f} (model: {current_best_model_id})")

        # Compare with best_overall_fitness and update if current is better
        best_overall_fitness = metadata.get('best_overall_fitness', 0.0)
        best_overall_uuid = metadata.get('best_overall_uuid', models_uuid)
        
        if current_best_fitness > best_overall_fitness:
            logger.info(f"New best overall fitness found: {current_best_fitness:.4f} > {best_overall_fitness:.4f}")
            best_overall_fitness = current_best_fitness
            best_overall_uuid = models_uuid
        else:
            logger.info(f"Current best fitness {current_best_fitness:.4f} does not exceed overall best {best_overall_fitness:.4f}")

        # Update fitness history
        fitness_history = metadata.get('fitness_history', [])
        fitness_history.append(current_best_fitness)
        logger.info(f"Fitness history updated: {fitness_history}")

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

        # Save selected models to database with a new valid UUID
        new_uuid = generate_uuid()
        


        # Copy metadata to the new population (increment generation)
        new_metadata = metadata.copy()
        new_metadata['generation'] = metadata['generation'] + 1
        new_metadata['uuid'] = new_uuid
        new_metadata['fitness_history'] = fitness_history
        new_metadata['best_overall_fitness'] = best_overall_fitness
        new_metadata['best_overall_uuid'] = best_overall_uuid
        
        # Save metadata for the new population
        save_population_with_metadata(
            population_uuid=new_uuid,
            generation=new_metadata['generation'],
            max_generations=new_metadata['max_generations'],
            fitness_threshold=new_metadata['fitness_threshold'],
            fitness_history=new_metadata['fitness_history'],
            best_overall_fitness=new_metadata['best_overall_fitness'],
            best_overall_uuid=new_metadata['best_overall_uuid'],
            original_params=new_metadata['original_params']
        )

        # Save each selected model to the new population
        i = 0
        for model_id, model_data in selected_models.items():
            save_model(new_uuid, str(i), model_data)
            i += 1


        logger.info(f"Selected {len(selected_models)} best models and saved to new population: {new_uuid}")
        logger.info(f"Population metadata copied with generation incremented to {new_metadata['generation']}")
        
        # Send metadata to continue-algorithm topic
        logger.info("🔄 Sending metadata to continue-algorithm topic...")
        producer = create_producer()
        
        continue_algorithm_data = {
            "uuid": new_uuid,
            "generation": new_metadata['generation'],
            "max_generations": new_metadata['max_generations'],
            "fitness_threshold": new_metadata['fitness_threshold'],
            "fitness_history": new_metadata['fitness_history'],
            "best_overall_fitness": new_metadata['best_overall_fitness'],
            "best_overall_uuid": new_metadata['best_overall_uuid'],
            "original_params": new_metadata['original_params']
        }
                
        # create descendant population
        for i in range(num_to_select):
            child_genome = cross_genomes_local(
                selected_models[random.choice(list(selected_models.keys()))],
                selected_models[random.choice(list(selected_models.keys()))]
            )
            child_genome = mutate_genome_local(child_genome)
            save_model(new_uuid, str(num_to_select+i+1), child_genome, score=0)
        
        logger.info(f"Created {num_to_select} children from selected models")
            
        continue_message = json.dumps(continue_algorithm_data)
        producer.produce("continue-algorithm", continue_message.encode('utf-8'))
        producer.flush()
        
        
        logger.info(f"✅ Sent children population {new_uuid} to continue-algorithm topic")
        return new_uuid, new_uuid

    except Exception as e:
        logger.error(f"Error in select_best_architectures: {e}")
        runtime_error_message(topic)
        return None, None