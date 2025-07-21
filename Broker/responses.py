import json
from confluent_kafka import Producer
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


def bad_optimizer_message(topic):
    """Send an invalid optimizer error to Kafka."""
    response = {'error': 'Invalid optimizer', 'status_code': 400}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)


def bad_model_message(topic, additional_info=None):
    """Send an invalid model error to Kafka."""
    response = {'error': 'Invalid model', 'status_code': 400}
    if additional_info:
        response['details'] = additional_info
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)


def runtime_error_message(topic, additional_info=None):
    """Send a runtime error message to Kafka."""
    response = {'error': 'Runtime error', 'status_code': 500}
    if additional_info:
        response['details'] = additional_info
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)


def response_message(topic, message, status_code):
    """Send a custom response message to Kafka."""
    response = {'message': message, 'status_code': status_code}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)


def bad_request_message(topic, message):
    """Send a bad request error message to Kafka."""
    response = {'error': 'Bad request', 'message': message, 'status_code': 400}
    logger.error(f"Response: {response}")
    send_kafka_response(topic, response)
