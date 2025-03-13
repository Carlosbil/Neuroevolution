import json
from utils import logger, get_possible_models, create_producer, create_consumer, produce_message
from responses import ok_message, bad_request_message, runtime_error_message, response_message
from confluent_kafka import KafkaError


def process_create_child(topic, data):
    """Process messages from the 'create_child' topic and send a response to Kafka."""
    try:
        logger.info("Processing create_child")
        if 'model_id' not in data or 'second_model_id' not in data:
            bad_request_message(topic, "model_id and second_model_id are required")
            return None

        model_id = str(data['model_id'])
        second_model_id = str(data['second_model_id'])
        possible_models = get_possible_models(data['uuid'])

        if model_id not in possible_models or second_model_id not in possible_models:
            bad_request_message(topic, "One or both models not found")
            return None

        json_to_send = {
            "1": possible_models[model_id],
            "2": possible_models[second_model_id]
        }
        
        # Send the message to the genome-create-child topic
        producer = create_producer()
        topic_to_send = "genome-create-child"
        response_topic = f"{topic_to_send}-response"
        produce_message(producer, topic_to_send, json.dumps(json_to_send), times=1)
        
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
        
        if response.get('status_code', 0) == 200:
            ok_message(topic, response.get('message', {}))
            return response.get('message', {})
        else:
            response_message(topic, response, response.get('status_code', 0))
            return None
    except Exception as e:
        logger.error(f"Error in create_child: {e}")
        runtime_error_message(topic)
        return None
