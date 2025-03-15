from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError


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
            
        # extract the models
        models_uuid = data["message"]['uuid']
        model_id = data["message"]['model_id']
        if data.get("status_code") == 200:
            path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
            if not os.path.exists(path):
                bad_model_message(topic_response)
                return None, None

            with open(path, 'r') as file:
                models = json.load(file)
            score = data.get("message", {}).get("score")
            if score is not None:
                logger.info(f"Model {model_id} evaluated with score {score}")
                models[model_id]["score"] = score 
            else:
                logger.error(f"Error evaluating model {model_id}: No score in response")
                models[model_id]["score"] = 0
        else:
            logger.error(f"Error evaluating model {model_id}: {data.get('status_code')}")
            models[model_id]["score"] = 0

        with open(path, 'w') as file:
            json.dump(models, file, indent=4)

        ok_message(topic_response, {"uuid": models_uuid, "path": path, "message": "Population evaluated successfully"})
        return models_uuid, path
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        runtime_error_message(topic_response)
        return None, None