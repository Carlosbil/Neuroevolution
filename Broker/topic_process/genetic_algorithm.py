from utils import logger, create_producer, produce_message, create_consumer, generate_uuid
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message, response_message
from database import get_population, population_exists, save_population, save_model, get_best_fitness_from_population
import requests
import json
import os
import random
import time
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


def create_children_from_selected(selected_uuid, population_size, mutation_rate=0.1):
    """Create children from selected models using crossover and mutation.
    
    Args:
        selected_uuid (str): UUID of the selected parent population
        population_size (int): Target population size for children
        mutation_rate (float): Mutation rate for genetic operations
        
    Returns:
        str: UUID of the new children population, or None on error
    """
    try:
        logger.info(f"Creating {population_size} children from selected models")
        
        # Get selected models from database
        if not population_exists(selected_uuid):
            logger.error(f"Selected population not found: {selected_uuid}")
            return None
            
        selected_models = get_population(selected_uuid)
        if not selected_models:
            logger.error(f"No models found in selected population: {selected_uuid}")
            return None
            
        selected_model_list = list(selected_models.values())
        if len(selected_model_list) < 2:
            logger.error("Need at least 2 models to create children")
            return None
            
        # Create new population for children
        children_uuid = f"{selected_uuid}_children"
        save_population(children_uuid)
        
        # Create children by crossing selected models
        for i in range(population_size):
            # Randomly select two parent models
            parent1_data = random.choice(selected_model_list)
            parent2_data = random.choice(selected_model_list)
            
            # Perform crossover - create new genome from the two parents
            child_data = cross_genomes_local(parent1_data, parent2_data)
            
            # Apply mutation - mutate the child genome in-place
            child_data = mutate_genome_local(child_data, mutation_rate)
            
            # Reset score to 0 for the new child (it hasn't been evaluated yet)
            child_data['score'] = 0
            
            # Save the child to the children population
            child_id = f"child_{i+1}"
            save_model(children_uuid, child_id, child_data)
        
        logger.info(f"Created {population_size} children and saved to population: {children_uuid}")
        return children_uuid
        
    except Exception as e:
        logger.error(f"Error creating children: {e}")
        return None


def cross_genomes_local(genome1, genome2):
    """Cross two genomes. 50% of the time, the gene will be taken from genome1, otherwise from genome2."""
    import torch
    logger.debug(f"Crossing genomes")
    new_genome = {}
    for key in genome1:
        new_genome[key] = genome1[key] if torch.rand(1) > 0.5 else genome2[key]
    return new_genome


def mutate_genome_local(genome, mutation_rate=0.1):
    """Mutate a genome. Randomly changes a gene with a given mutation rate."""
    import torch
    
    # Maps from select_best_architectures
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
    
    for key in genome:
        if torch.rand(1) < mutation_rate:
            logger.debug(f"Mutating key: {key} with value: {genome[key]}")
            
            if key in ['Number of Convolutional Layers', 'Number of Fully Connected Layers']:
                genome[key] = torch.randint(1, 5, (1,)).item()
            
            elif key == 'Number of Nodes in Each Layer':
                genome[key] = [torch.randint(16, 128, (1,)).item() for _ in range(genome['Number of Fully Connected Layers'])]
            
            elif key == 'Activation Functions':
                activation_function_key = random.choice(list(MAP_ACTIVATE_FUNCTIONS.keys()))
                genome[key] = [activation_function_key for _ in range(genome['Number of Fully Connected Layers'])]
            
            elif key in ['Dropout Rate', 'Learning Rate']:
                factor = random.choice([0.1, 0.01, 0.001])
                genome[key] *= factor if random.random() < 0.5 else 1 / factor
                genome[key] = min(round(genome[key], 4), 0.9)
            
            elif key == 'Batch Size':
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'Optimizer':
                optimizer_key = random.choice(list(MAP_OPTIMIZERS.keys()))
                genome[key] = optimizer_key
            
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


def check_convergence_criteria(generation, max_generations, best_fitness, fitness_threshold, fitness_history):
    """Check if the convergence criteria have been met.
    
    Args:
        generation (int): Current generation number
        max_generations (int): Maximum number of generations
        best_fitness (float): Best fitness score in current generation
        fitness_threshold (float): Target fitness threshold
        fitness_history (list): History of best fitness scores
        
    Returns:
        tuple: (converged, reason)
    """
    try:
        # Check maximum generations
        if generation >= max_generations:
            return True, f"Maximum generations reached ({max_generations})"
        
        # Check fitness threshold
        if best_fitness >= fitness_threshold:
            return True, f"Fitness threshold reached ({best_fitness:.4f} >= {fitness_threshold})"
        
        # Check for fitness stagnation (no improvement in last 5 generations)
        if len(fitness_history) >= 5:
            recent_fitness = fitness_history[-5:]
            if all(abs(f - recent_fitness[0]) < 0.001 for f in recent_fitness):
                return True, f"Fitness stagnation detected (no improvement in last 5 generations)"
        
        return False, "Convergence criteria not met"
        
    except Exception as e:
        logger.error(f"Error checking convergence criteria: {e}")
        return False, f"Error checking convergence: {e}"


def get_best_fitness_from_models(models_uuid):
    """Get the best fitness score from a population of models using the database.
    
    Args:
        models_uuid (str): UUID of the models population
        
    Returns:
        float: Best fitness score, or 0.0 if error
    """
    try:
        # Use the database function directly
        return get_best_fitness_from_population(models_uuid)
        
    except Exception as e:
        logger.error(f"Error getting best fitness: {e}")
        return 0.0


def process_genetic_algorithm(topic, data):
    """Process the complete genetic algorithm with multiple generations.
    
    This function implements the full evolutionary loop:
    1. Create initial population
    2. For each generation:
       a. Evaluate population
       b. Select best individuals
       c. Create children through crossover and mutation
       d. Replace population with children
       e. Check convergence criteria
    3. Return best model and statistics
    """
    try:
        logger.info("🧬 Starting complete genetic algorithm with multiple generations")
        
        # Extract parameters
        max_generations = data.get('max_generations', 10)
        fitness_threshold = data.get('fitness_threshold', 0.95)
        population_size = data.get('num_poblation', 10)
        
        logger.info(f"🎯 Parameters: max_generations={max_generations}, fitness_threshold={fitness_threshold}, population_size={population_size}")
        
        # Step 1: Create initial population
        logger.info("🎲 Creating initial population...")
        models_uuid, path = process_create_initial_population(TOPIC_PROCESSORS["create-initial-population"], data)
        if models_uuid is None:
            return response_message(topic, "Error creating initial population", 500)
        
        # Initialize tracking variables
        generation = 0
        fitness_history = []
        best_overall_fitness = 0.0
        best_overall_uuid = models_uuid
        
        # Main evolutionary loop
        while generation < max_generations:
            logger.info(f"🔄 Generation {generation + 1}/{max_generations}")
            
            # Step 2: Evaluate population
            logger.info(f"📊 Evaluating population...")
            json_to_send = {"uuid": models_uuid}
            evaluated_uuid, evaluated_path = process_evaluate_population(TOPIC_PROCESSORS["evaluate-population"], json_to_send)
            if evaluated_uuid is None:
                return response_message(topic, f"Error evaluating population in generation {generation + 1}", 500)
            
            # Get best fitness for this generation
            best_fitness = get_best_fitness_from_models(evaluated_uuid)
            fitness_history.append(best_fitness)
            
            # Update best overall
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_uuid = evaluated_uuid
            
            logger.info(f"🏆 Generation {generation + 1} best fitness: {best_fitness:.4f} (overall best: {best_overall_fitness:.4f})")
            
            # Step 3: Check convergence criteria
            converged, reason = check_convergence_criteria(generation + 1, max_generations, best_fitness, fitness_threshold, fitness_history)
            if converged:
                logger.info(f"✅ Convergence achieved: {reason}")
                break
            
            # Step 4: Select best architectures
            logger.info(f"🎯 Selecting best architectures...")
            json_to_send = {"uuid": evaluated_uuid}
            selected_uuid, selected_path = process_select_best_architectures(TOPIC_PROCESSORS["select-best-architectures"], json_to_send)
            if selected_uuid is None:
                return response_message(topic, f"Error selecting best architectures in generation {generation + 1}", 500)
            
            # Step 5: Create children (next generation)
            logger.info(f"👶 Creating children for next generation...")
            children_uuid = create_children_from_selected(selected_uuid, population_size, mutation_rate=data.get('mutation_rate', 0.1))
            if children_uuid is None:
                return response_message(topic, f"Error creating children in generation {generation + 1}", 500)
            
            # Step 6: Use children as next generation (they become the new population)
            logger.info(f"🔄 Using children as next generation...")
            models_uuid = children_uuid
            generation += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
        
        # Final results
        logger.info(f"🎉 Genetic algorithm completed!")
        logger.info(f"📈 Total generations: {generation}")
        logger.info(f"🏆 Best fitness achieved: {best_overall_fitness:.4f}")
        logger.info(f"📊 Fitness history: {fitness_history}")
        
        return ok_message(topic, {
            "uuid": best_overall_uuid,
            "path": path,
            "generations_completed": generation,
            "best_fitness": best_overall_fitness,
            "fitness_history": fitness_history,
            "convergence_reason": reason if converged else "Maximum generations reached",
            "message": f"Genetic algorithm completed successfully: {generation} generations, best fitness {best_overall_fitness:.4f}"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in genetic algorithm: {e}")
        return runtime_error_message(topic)