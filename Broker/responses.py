from flask import jsonify
from utils import logger
def ok_message(message):
    response = jsonify({'message': message})
    response.status_code = 200
    logger.info(f"Response: {response}")
    return response

def bad_optimizer_message():
    response = jsonify({'error': 'Invalid optimizer'})
    response.status_code = 400
    logger.error(f"Response: {response}")
    return response

def bad_model_message():
    response = jsonify({'error': 'Invalid model'})
    response.status_code = 400
    logger.error(f"Response: {response}")
    return response

def runtime_error_message():
    response = jsonify({'error': 'Runtime error'})
    response.status_code = 500
    logger.error(f"Response: {response}")
    return response

def response_message(message, status_code):
    response = jsonify(message)
    response.status_code = status_code
    logger.error(f"Response: {response}")
    return response