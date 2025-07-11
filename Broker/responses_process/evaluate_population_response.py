from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError
from database import update_model_score, get_population, population_exists, save_model
from topic_process.select_best_architectures import process_select_best_architectures


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
                    
                    # Check if all models have been evaluated with non-zero scores
                    updated_models = get_population(models_uuid)
                    if check_how_many(updated_models):
                        logger.info(f"🥰 All models in population {models_uuid} have been evaluated. Calling select_best_architectures directly.")
                        # Call process_select_best_architectures directly without Kafka
                        process_select_best_architectures("select-best-architectures", {"uuid": models_uuid})
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
                models[model_id]["score"] = -1
                update_model_score(models_uuid, model_id, -1)

        ok_message(topic_response, {"uuid": models_uuid, "message": "Population evaluated successfully"})
        return models_uuid, None
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        runtime_error_message(topic_response)
        return None, None
    

def check_how_many(models):
    """
    Review if we have got all scores for models in a population.
    Check if all individuals have a non-zero score value.
    
    :param models: Dictionary of models with their data
    :type models: dict
    :return: True if all models have non-zero scores, False otherwise
    :rtype: bool
    """
    if not models:
        return False
    
    for model_data in models.values():
        if "score" not in model_data or model_data["score"] is None or model_data["score"] == 0:
            return False
    
    return True
    