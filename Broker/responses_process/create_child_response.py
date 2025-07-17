import json
from utils import logger, generate_uuid, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message, bad_request_message
from confluent_kafka import KafkaError
from database import save_population, save_model, get_population, population_exists, get_population_metadata, save_population_with_metadata

def process_create_child_response(topic, response):
    """
    Process messages from the 'genome-create-child-response' topic
    and send an appropriate response back to Kafka.
    """
    try:
        topic_response = "create-child"
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
        
        # Get population metadata from original population
        logger.info(f"Getting population metadata for {models_uuid}")
        metadata = get_population_metadata(models_uuid)
        
        if not metadata:
            logger.warning(f"No metadata found for population {models_uuid}, using defaults")
            metadata = {
                'generation': 0,
                'max_generations': 10,
                'fitness_threshold': 0.95,
                'fitness_history': [],
                'best_overall_fitness': 0.0,
                'best_overall_uuid': models_uuid,
                'original_params': {}
            }
        
        # Save population to database with metadata
        save_population_with_metadata(
            population_uuid=new_models_uuid,
            generation=metadata['generation'],
            max_generations=metadata['max_generations'],
            fitness_threshold=metadata['fitness_threshold'],
            fitness_history=metadata['fitness_history'],
            best_overall_fitness=metadata['best_overall_fitness'],
            best_overall_uuid=metadata['best_overall_uuid'],
            original_params=metadata['original_params']
        )
        
        # Save each model to database
        for model_id, model_data in children.items():
            save_model(new_models_uuid, model_id, model_data)

        logger.info(f"✅ Created children population {new_models_uuid} with {len(children)} models")
        logger.info(f"📊 Population metadata copied with generation {metadata['generation']}")
        
        # Send metadata to continue-algorithm topic
        logger.info("🔄 Sending metadata to continue-algorithm topic...")
        producer = create_producer()
        
        continue_algorithm_data = {
            "uuid": new_models_uuid,
            "generation": metadata['generation'],
            "max_generations": metadata['max_generations'],
            "fitness_threshold": metadata['fitness_threshold'],
            "fitness_history": metadata['fitness_history'],
            "best_overall_fitness": metadata['best_overall_fitness'],
            "best_overall_uuid": metadata['best_overall_uuid'],
            "original_params": metadata['original_params']
        }
        
        continue_message = json.dumps(continue_algorithm_data)
        producer.produce("continue-algorithm", continue_message.encode('utf-8'))
        producer.flush()
        
        logger.info(f"✅ Sent children population {new_models_uuid} to continue-algorithm topic")

        # Build and send the success response message
        result_message = {
            'uuid': new_models_uuid,
            'generation': metadata['generation'],
            'metadata': metadata
        }
        ok_message(topic_response, result_message)
        
        return new_models_uuid, None

    except Exception as e:
        # Log and return runtime error in case of unexpected exception
        logger.error(f"Error in create_child_response: {e}")
        return runtime_error_message(topic_response)
