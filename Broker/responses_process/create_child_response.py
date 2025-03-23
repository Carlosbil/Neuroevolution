import os
import json
from utils import logger, generate_uuid, create_producer, create_consumer, produce_message, get_storage_path
from responses import ok_message, bad_model_message, runtime_error_message, response_message, bad_request_message
from confluent_kafka import KafkaError
from utils import get_storage_path

def process_create_child_response(topic, response):
    """
    Process messages from the 'genome-create-child-response' topic
    and send an appropriate response back to Kafka.
    """
    try:
        topic_response = "create-child"

        # Check if the response status is OK (200)
        if response.get('status_code', 0) == 200:
            data = response.get('message', {})
            children = data.get('children', {})
            topic_response = "create-child"
            logger.info("Processing create_child_response")

            # Ensure 'uuid' is present in the response
            if 'uuid' not in data:
                bad_request_message(topic, "uuid is required")
                return None, None

            # Load models from the file system based on the provided UUID
            models_uuid = data['uuid']
            path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
            if not os.path.exists(path):
                bad_model_message(topic)
                return None, None

            with open(path, 'r') as file:
                models = json.load(file)

            new_position = len(children)

            # If no models are found, return a bad request message
            if models == {}:
                bad_request_message(topic, "No models found")

            # Convert the dictionary values to a list to iterate through the original models
            original_models = list(models.values())

            # Append each original model to the 'children' dictionary with a new index
            for model in original_models:
                children[str(new_position)] = model
                new_position += 1

            # Generate a new UUID for the updated model set
            models_uuid = generate_uuid()

            # Save the updated models (now including the new children) to a new file
            path = os.path.join(get_storage_path(), f'{models_uuid}.json')
            with open(path, 'w') as file:
                json.dump(children, file)

            # Build and send the success response message
            result_message = {
                'uuid': models_uuid,
            }
            ok_message(topic_response, result_message)
            return models_uuid, path

    except Exception as e:
        # Log and return runtime error in case of unexpected exception
        logger.error(f"Error in create_child_response: {e}")
        return runtime_error_message(topic_response)
