from utils import logger, create_producer, produce_message, create_consumer, generate_uuid
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message, response_message
from database import get_population, population_exists, get_best_fitness_from_population
import json

def process_check_population(topic, data):
    """Check if a population exists in the database."""
    try:
        logger.info(f"Checking population existence with data: {data}")
        
        # Validate input data
        if 'uuid' not in data:
            logger.error("Missing required field 'uuid' in check_population data")
            bad_request_message(topic, "uuid is required")
            return None, None
        
        models_uuid = data['uuid']
        
        # Check if population exists in database
        if population_exists(models_uuid):
            logger.info(f"Population {models_uuid} exists in database")
            return ok_message(topic, {"exists": True, "uuid": models_uuid})
        else:
            logger.info(f"Population {models_uuid} does not exist in database")
            return ok_message(topic, {"exists": False, "uuid": models_uuid})
            
    except Exception as e:
        logger.error(f"Error in check_population: {e}")
        return runtime_error_message(topic, f"Error checking population: {str(e)}")

def process_check_evaluation(topic, data):
    """Check if population evaluation is complete."""
    try:
        logger.info(f"Checking evaluation status with data: {data}")
        
        # Validate input data
        if 'uuid' not in data:
            logger.error("Missing required field 'uuid' in check_evaluation data")
            bad_request_message(topic, "uuid is required")
            return None, None
        
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
        
        # Check if all models have been evaluated (score > 0)
        evaluated_count = 0
        total_count = len(models)
        
        for model_id, model_data in models.items():
            if isinstance(model_data, dict) and model_data.get('score', 0) > 0:
                evaluated_count += 1
        
        evaluation_complete = evaluated_count == total_count
        
        logger.info(f"Evaluation status for {models_uuid}: {evaluated_count}/{total_count} models evaluated")
        
        return ok_message(topic, {
            "evaluation_complete": evaluation_complete,
            "evaluated_count": evaluated_count,
            "total_count": total_count,
            "uuid": models_uuid
        })
        
    except Exception as e:
        logger.error(f"Error in check_evaluation: {e}")
        return runtime_error_message(topic, f"Error checking evaluation: {str(e)}")

def process_get_best_fitness(topic, data):
    """Get the best fitness from a population."""
    try:
        logger.info(f"Getting best fitness with data: {data}")
        
        # Validate input data
        if 'uuid' not in data:
            logger.error("Missing required field 'uuid' in get_best_fitness data")
            bad_request_message(topic, "uuid is required")
            return None, None
        
        models_uuid = data['uuid']
        
        # Check if population exists in database
        if not population_exists(models_uuid):
            logger.error(f"Population not found in database: {models_uuid}")
            bad_model_message(topic, f"Population not found: {models_uuid}")
            return None, None
        
        # Get best fitness from database
        best_fitness = get_best_fitness_from_population(models_uuid)
        
        logger.info(f"Best fitness for {models_uuid}: {best_fitness}")
        
        return ok_message(topic, {
            "best_fitness": best_fitness,
            "uuid": models_uuid
        })
        
    except Exception as e:
        logger.error(f"Error in get_best_fitness: {e}")
        return runtime_error_message(topic, f"Error getting best fitness: {str(e)}")
