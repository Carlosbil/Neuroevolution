from utils import logger, get_possible_models
from responses import ok_message, bad_request_message, runtime_error_message, response_message
import requests


def process_create_child(topic, data):
    """Process messages from the 'create_child' topic and send a response to Kafka."""
    try:
        logger.info("Processing create_child")
        if 'model_id' not in data or 'second_model_id' not in data:
            return bad_request_message(topic, "model_id and second_model_id are required")

        model_id = str(data['model_id'])
        second_model_id = str(data['second_model_id'])
        possible_models = get_possible_models(data['uuid'])

        if model_id not in possible_models or second_model_id not in possible_models:
            return bad_request_message(topic, "One or both models not found")

        json_to_send = {
            "1": possible_models[model_id],
            "2": possible_models[second_model_id]
        }
        server_url = "http://127.0.0.1:5002/create_child"

        response = requests.post(server_url, json=json_to_send)
        if response.status_code == 200:
            return ok_message(topic, response.json())
        else:
            return response_message(topic, response.json(), response.status_code)
    except Exception as e:
        logger.error(f"Error in create_child: {e}")
        return runtime_error_message(topic)
