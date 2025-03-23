from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError
from utils import get_storage_path
from database import update_model_score, get_population, population_exists, save_model


def process_evaluate_population_response(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    topic_response = "evaluate-population"
    try:
        logger.info("Processing evaluate_population")
        if data.get("message"):
            if 'uuid' not in data["message"]:
                bad_request_message(topic_response, "uuid is required")
                return None, None
            
            if not 'model_id' in data["message"]:
                bad_request_message(topic_response, "model_id is required")
                return None, None
            
        # extract the models
        models_uuid = data["message"]['uuid']
        model_id = data["message"]['model_id']
        
        # Check if population exists in database
        if not population_exists(models_uuid):
            bad_model_message(topic_response)
            return None, None
            
        if data.get("status_code") == 200:
            # Get models from database
            models = get_population(models_uuid)
            score = data.get("message", {}).get("score")
            
            if score is not None:
                logger.info(f"Model {model_id} evaluated with score {score}")
                # Update score in database
                if model_id in models:
                    models[model_id]["score"] = score
                    update_model_score(models_uuid, model_id, score)
                else:
                    logger.error(f"Model {model_id} not found in population {models_uuid}")
            else:
                logger.error(f"Error evaluating model {model_id}: No score in response")
                if model_id in models:
                    models[model_id]["score"] = 0
                    update_model_score(models_uuid, model_id, 0)
        else:
            logger.error(f"Error evaluating model {model_id}: {data.get('status_code')}")
            # Get models from database to ensure model_id exists
            models = get_population(models_uuid)
            if model_id in models:
                models[model_id]["score"] = 0
                update_model_score(models_uuid, model_id, 0)

        ok_message(topic_response, {"uuid": models_uuid, "message": "Population evaluated successfully"})
        return models_uuid, None
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        runtime_error_message(topic_response)
        return None, None
    

def check_how_many(models):
    """
    Review if we have got all scores for models in a population.
    
    :param models: Dictionary of models with their data
    :type models: dict
    :return: Number of models with scores and total number of models
    :rtype: tuple(int, int)
    """
    scored_models = sum(1 for model_data in models.values() if "score" in model_data and model_data["score"] is not None)
    return scored_models, len(models)
    