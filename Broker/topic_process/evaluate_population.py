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
            return bad_request_message(topic, "uuid is required")
        
        # extract the models
        models_uuid = data['uuid']
        path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{models_uuid}.json')
        if not os.path.exists(path):
            return bad_model_message(topic)

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
            try:
                # Now we wait for the response
                consumer = create_consumer()
                consumer.subscribe([response_topic])
                response = None
                while response is None:
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            logger.info(f"Reached end of partition: {msg.topic()} [{msg.partition()}]")
                        else:
                            logger.error(f"Kafka error: {msg.error()}")
                        continue
                    response = json.loads(msg.value().decode('utf-8'))
                consumer.close()
                
                # If the response is successful, update the model with the score
                if response.get("status_code") == 200:
                    score = response.get("message", {}).get("score")
                    if score is not None:
                        logger.info(f"Model {model_id} evaluated with score {score}")
                        models[model_id]["score"] = score 
                    else:
                        logger.error(f"Error evaluating model {model_id}: No score in response")
                        models[model_id]["score"] = 0
                else:
                    logger.error(f"Error evaluating model {model_id}: {response.get('status_code')}")
                    models[model_id]["score"] = 0
            except Exception as e:
                logger.error(f"Error evaluating model {model_id}: {e}")
                models[model_id]["score"] = f"Error: {str(e)}"

        with open(path, 'w') as file:
            json.dump(models, file, indent=4)

        return ok_message(topic, {"uuid": models_uuid, "path": path, "message": "Population evaluated successfully"})
    except Exception as e:
        logger.error(f"Error in evaluate_population: {e}")
        return runtime_error_message(topic)