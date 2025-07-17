import json
from utils import logger, create_producer

def send_kafka_response(topic, message):
    """Send a response message to a Kafka topic."""
    producer = create_producer()
    response_topic = f"{topic}-response"
    producer.produce(response_topic, json.dumps(message).encode('utf-8'))
    producer.flush()
    logger.info(f"Sent response to topic '{response_topic}': {message}")

def ok_message(topic, message):
    """Send a success response to Kafka."""
    response = {'message': message, 'status_code': 200}
    logger.info(f"Response: {response}")
    send_kafka_response(topic, response)
    return response

def bad_model_message(topic, error_message="Invalid model"):
    """Send an invalid model error to Kafka."""
    response = {'error': error_message, 'status_code': 400}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)
    return response

def runtime_error_message(topic, error_message="Runtime error"):
    """Send a runtime error to Kafka."""
    response = {'error': error_message, 'status_code': 500}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)
    return response

def bad_request_message(topic, error_message="Bad request"):
    """Send a bad request error to Kafka."""
    response = {'error': error_message, 'status_code': 400}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)
    return response

def response_message(topic, message, status_code):
    """Send a response message with custom status code to Kafka."""
    response = {'message': message, 'status_code': status_code}
    if status_code >= 400:
        logger.error(f"Response: {response}")
    else:
        logger.info(f"Response: {response}")
    send_kafka_response(topic, response)
    return response
