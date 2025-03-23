import os
import json
import requests
from utils import logger, generate_uuid, check_initial_poblation, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message
from confluent_kafka import KafkaError
from utils import get_storage_path
def process_create_initial_population_response(topic, response):
    """Process messages from the 'create_initial_population' topic and send a response to Kafka."""
    try:
        topic_response = "create-initial-population"
        if response.get('status_code', 0) == 200:
            message = response.get('message', {})
            models = message.get('models', {})
            models_uuid = generate_uuid()
            # Use the configurable storage path
            
            path = os.path.join(get_storage_path(), f'{models_uuid}.json')

            with open(path, 'w') as file:
                json.dump(models, file)
            message = {
                "uuid": models_uuid,
            }
            ok_message(topic_response, message)
            
            # now go to step_2, evaluate_population
            topic_to_send = "evaluate-population"
            data = {
                "uuid": models_uuid,
                "path": path
            }
            message = json.dumps(data)
            producer = create_producer()
            produce_message(producer, topic_to_send, data, 1)
            return
        else:
            return response_message(topic_response, response, response.get('status_code', 0))
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        return runtime_error_message(topic_response)