import os
import json
from utils import logger, generate_uuid, create_producer, create_consumer, produce_message, get_storage_path
from responses import ok_message, bad_model_message, runtime_error_message, response_message, bad_request_message
from confluent_kafka import KafkaError
from utils import get_storage_path

def process_create_child_response(topic, response):
    """Process messages from the 'genome-create-child-response' topic and send a response to Kafka."""
    try:
        topic_response = "create-child"
        if response.get('status_code', 0) == 200:
            data = response.get('message', {})
            models = data.get('models', {})
            topic_response = "create-child"
            logger.info("Processing create_child_response")
            
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
            
            new_position = len(models)
            if models == {}:
                bad_request_message(topic, "No models found")
                
            for model in models:
                models[new_position] = model
                new_position += 1
            
            # create a new uuid
            models_uuid = generate_uuid()
            # Save the models
            path = os.path.join(get_storage_path(), f'{models_uuid}.json')
            with open(path, 'w') as file:
                json.dump(models, file)
            
            result_message = {
                'uuid': models_uuid,
            }
            ok_message(topic_response, result_message)
            return models_uuid, path
 
    except Exception as e:
        logger.error(f"Error in create_child_response: {e}")
        return runtime_error_message(topic_response)