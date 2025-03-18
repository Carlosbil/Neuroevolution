from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError


def process_evaluate_population(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing evaluate_population")
        if 'uuid' not in data:
            bad_request_message(topic, "uuid is required")
            return None, None
        
        # extract the models
        models_uuid = data['uuid']
        path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            bad_model_message(topic)
            return None, None

        with open(path, 'r') as file:
            models = json.load(file)

        # For each model, send a message to the evolutioner-create-cnn-model topic
        for model_id, model_data in models.items():
            # extract the model
            json_to_send = model_data
            json_to_send["model_id"] = model_id
            json_to_send["uuid"] = models_uuid
            producer = create_producer()
            topic_to_sed = "evolutioner-create-cnn-model"
            response_topic = f"{topic_to_sed}-response"
            produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)

        return
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        runtime_error_message(topic)
        return None, None