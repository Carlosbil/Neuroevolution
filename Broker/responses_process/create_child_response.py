import json
from utils import logger, generate_uuid, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message, bad_request_message
from confluent_kafka import KafkaError
from database import save_population, save_model, get_population, population_exists

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

            # Load models from the database based on the provided UUID
            models_uuid = data['uuid']
            
            # Check if population exists in database
            if not population_exists(models_uuid):
                bad_model_message(topic)
                return None, None

            # Get models from database
            models = get_population(models_uuid)

            new_position = len(children)

            # If no models are found, return a bad request message
            if models == {}:
                bad_request_message(topic, "No models found")
                return None, None

            # Convert the dictionary values to a list to iterate through the original models
            original_models = list(models.values())

            # Append each original model to the 'children' dictionary with a new index
            for model in original_models:
                children[str(new_position)] = model
                new_position += 1

            # Generate a new UUID for the updated model set
            new_models_uuid = generate_uuid()
            
            # Save population to database
            save_population(new_models_uuid)
            
            # Save each model to database
            for model_id, model_data in children.items():
                save_model(new_models_uuid, model_id, model_data)

            # Build and send the success response message
            result_message = {
                'uuid': new_models_uuid,
            }
            ok_message(topic_response, result_message)
            
            return new_models_uuid, None

    except Exception as e:
        # Log and return runtime error in case of unexpected exception
        logger.error(f"Error in create_child_response: {e}")
        return runtime_error_message(topic_response)
