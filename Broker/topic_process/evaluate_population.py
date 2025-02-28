from utils import logger
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os

def process_evaluate_population(topic, data):
    """Process messages from the 'evaluate_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing evaluate_population")
        if 'uuid' not in data:
            return bad_request_message(topic, "uuid is required")

        models_uuid = data['uuid']
        path = os.path.join(os.path.dirname(__file__), 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            return bad_model_message(topic)

        with open(path, 'r') as file:
            models = json.load(file)

        server_url = "http://127.0.0.1:5000/create_cnn_model"
        for model_id, model_data in models.items():
            json_to_send = model_data
            json_to_send["model_id"] = model_id
            json_to_send["uuid"] = models_uuid

            try:
                response = requests.post(server_url, json=json_to_send)
                if response.status_code == 200:
                    json_response = response.json()
                    score = json_response.get("message", {}).get("score")
                    models[model_id]["score"] = score if score is not None else "Error: No score returned"
                else:
                    models[model_id]["score"] = f"Error: {response.status_code}"
            except Exception as e:
                logger.error(f"Error evaluating model {model_id}: {e}")
                models[model_id]["score"] = f"Error: {str(e)}"

        with open(path, 'w') as file:
            json.dump(models, file, indent=4)

        return ok_message(topic, {"uuid": models_uuid, "path": path, "message": "Population evaluated successfully"})
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        return runtime_error_message(topic)