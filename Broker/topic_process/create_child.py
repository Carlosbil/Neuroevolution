import os
import json
from utils import logger, get_possible_models, create_producer, create_consumer, produce_message
from responses import ok_message, bad_request_message, runtime_error_message, response_message
from confluent_kafka import KafkaError
from utils import get_storage_path


def process_create_child(topic, data):
    """Process messages from the 'create_child' topic and send a response to Kafka."""
    try:
        logger.info("Processing create_child message")
        if 'uuid' not in data:
            bad_request_message(topic, "uuid is required")
            return None, None

        if not 'dataset_params' in data:
            bad_request_message(topic, "dataset_params is required")
            return None, None
        
        dataset_params = data['dataset_params']
        # Extract the models
        # extract the models
        models_uuid = data['uuid']
        path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            bad_model_message(topic)
            return None, None

        with open(path, 'r') as file:
            models = json.load(file)
            json_to_send = {
                "models": models,
                "uuid": models_uuid,
                "dataset_params": dataset_params
            }
            producer = create_producer()
            topic_to_send = "genome-create-child"
            response_topic = f"{topic_to_send}-response"
            produce_message(producer, topic_to_send, json.dumps(json_to_send), times=1)
        return 
    except Exception as e:
        logger.error(f"Error in create_child: {e}")
        runtime_error_message(topic)
        return
