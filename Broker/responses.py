from flask import jsonify

def ok_message(message):
    response = jsonify({'message': message})
    response.status_code = 200
    return response

def bad_optimizer_message():
    response = jsonify({'error': 'Invalid optimizer'})
    response.status_code = 400
    return response

def bad_model_message():
    response = jsonify({'error': 'Invalid model'})
    response.status_code = 400
    return response

def runtime_error_message():
    response = jsonify({'error': 'Runtime error'})
    response.status_code = 500
    return response

def response_message(message, status_code):
    response = jsonify(message)
    response.status_code = status_code
    return response