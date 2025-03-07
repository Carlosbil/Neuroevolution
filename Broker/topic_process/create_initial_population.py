import os
import json
import requests
from utils import logger, generate_uuid, check_initial_poblation, create_producer, create_consumer, produce_message
from responses import ok_message, bad_model_message, runtime_error_message, response_message
from confluent_kafka import KafkaError

def process_create_initial_population(topic, data):
    """Process messages from the 'create_initial_population' topic and send a response to Kafka."""
    try:
        logger.info("Processing create_initial_population")
        if not check_initial_poblation(data):
            bad_model_message(topic)
            return None, None

        # Send the message to the genome-create-initial-population topic
        json_to_send = data
        producer = create_producer()
        topic_to_sed = "genome-create-initial-population"
        response_topic = f"{topic_to_sed}-response"
        produce_message(producer, topic_to_sed, json.dumps(json_to_send), times=1)
        
        #Now we wait to the response
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
        
        # get the models
        if response.get('status_code', 0) == 200:
            message = response.get('message', {})
            models = message.get('models', {})
            models_uuid = generate_uuid()
            path = os.path.join(os.path.dirname(__file__),'..', 'models', f'{models_uuid}.json')

            with open(path, 'w') as file:
                json.dump(models, file)
            message = {
                "uuid": models_uuid,
                "path": path
            }
            ok_message(topic, message)
            return models_uuid, path
        else:
            return response_message(topic, response, response.get('status_code', 0))
    except Exception as e:
        logger.error(f"Error in create_initial_population: {e}")
        return runtime_error_message(topic)