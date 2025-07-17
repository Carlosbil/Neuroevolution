from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message, response_message
import json
import time
import requests


def handle_genetic_algorithm(topic, data):
    """Handle the first part of genetic algorithm: initial population creation and first generation evaluation.
    
    This function:
    1. Creates initial population
    2. Evaluates the population
    3. Sends the evaluated population to the next-generation topic
    """
    try:
        logger.info("🧬 Starting genetic algorithm - First generation processing")
        
        # Extract parameters
        max_generations = data.get('max_generations', 10)
        fitness_threshold = data.get('fitness_threshold', 0.95)
        population_size = data.get('num_population', 10)
        
        logger.info(f"🎯 Parameters: max_generations={max_generations}, fitness_threshold={fitness_threshold}, population_size={population_size}")
        
        # Step 1: Create initial population
        logger.info("🎲 Creating initial population...")
        create_initial_data = data
        
        # Send to broker to create initial population
        producer = create_producer()
        produce_message(producer, "create-initial-population", json.dumps(create_initial_data))
        logger.info("✅ Initial population creation request sent")
        producer.flush()
        return None, None
        
    except Exception as e:
        logger.error(f"❌ Error in genetic algorithm first generation: {e}")
        return runtime_error_message(topic)


def handle_continue_algorithm(topic, data):
    """Handle the continuation of genetic algorithm: check convergence and continue evaluation.
    
    This function:
    1. Receives selected population with metadata
    2. Checks convergence criteria (max generations, fitness threshold)
    3. If not converged, sends population to evaluate-population
    4. If converged, returns final results
    """
    try:
        logger.info("🔄 Continuing genetic algorithm - Checking convergence and evaluating")
        
        # Extract parameters from the data received from select-best-architectures
        uuid = data.get('uuid')
        generation = data.get('generation', 1)
        max_generations = data.get('max_generations', 10)
        fitness_threshold = data.get('fitness_threshold', 0.95)
        fitness_history = data.get('fitness_history', [])
        best_overall_fitness = data.get('best_overall_fitness', 0.0)
        best_overall_uuid = data.get('best_overall_uuid', uuid)
        original_params = data.get('original_params', {})
        
        if not uuid:
            logger.error("❌ No UUID provided for continuation")
            return bad_request_message(topic)
        
        logger.info(f"🎯 Processing generation {generation}/{max_generations} for UUID: {uuid}")
        logger.info(f"� Current best fitness: {best_overall_fitness:.4f}")
        
        # Check convergence criteria
        converged, reason = check_convergence_criteria(generation, max_generations, best_overall_fitness, fitness_threshold, fitness_history)
        
        if converged:
            logger.info(f"✅ Convergence achieved: {reason}")
            return ok_message(topic, {
                "uuid": best_overall_uuid,
                "generation": generation,
                "converged": True,
                "convergence_reason": reason,
                "best_fitness": best_overall_fitness,
                "fitness_history": fitness_history,
                "message": f"Genetic algorithm completed: {reason}"
            })
        
        # If not converged, send the selected population to evaluate-population
        logger.info(f"📊 Convergence not achieved, evaluating selected population {uuid}...")
        
        producer = create_producer()
        evaluate_data = {
            "uuid": uuid
        }
        
        produce_message(producer, "evaluate-population", json.dumps(evaluate_data))
        logger.info("✅ Population evaluation request sent")
        
        producer.flush()
        
        return ok_message(topic, {
            "uuid": uuid,
            "generation": generation,
            "converged": False,
            "best_fitness": best_overall_fitness,
            "message": f"Generation {generation} selected population sent for evaluation"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in genetic algorithm continuation: {e}")
        return runtime_error_message(topic)


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
