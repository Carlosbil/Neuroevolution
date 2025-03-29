import json
import requests
from utils import logger, generate_uuid, check_initial_poblation, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message
from confluent_kafka import KafkaError
from database import save_population, save_model
def process_create_initial_population_response(topic, response):
    """Process messages from the 'create_initial_population' topic and send a response to Kafka."""
    try:
        topic_response = "create-initial-population"
        if response.get('status_code', 0) == 200:
            message = response.get('message', {})
            models = message.get('models', {})
            models_uuid = generate_uuid()
            
            # Save population to database
            save_population(models_uuid)
            
            # Save each model to database
            for model_id, model_data in models.items():
                save_model(models_uuid, model_id, model_data)
                
            message = {
                "uuid": models_uuid,
            }
            ok_message(topic_response, message)
            
            # now go to step_2, evaluate_population
            topic_to_send = "evaluate-population"
            data = {
                "uuid": models_uuid
            }
            # Convert the dictionary to a JSON string before passing to produce_message
            message = json.dumps(data)
            producer = create_producer()
            produce_message(producer, topic_to_send, message, 1)
            return
        else:
            return response_message(topic_response, response, response.get('status_code', 0))
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        return runtime_error_message(topic_response)