# endpoints.py
from flask_restx import Resource
from flask import request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger, check_initial_poblation, generate_uuid, get_possible_models
from swagger_models import api, cnn_model_parameters, child_model_parameters, initial_poblation
from responses import ok_message, bad_model_message, bad_optimizer_message, runtime_error_message, response_message, bad_request_message
import requests
import json
import os

ns = api.namespace('', description='CNN Model operations')

@ns.route('/create_cnn_model')
class CNNModel(Resource):
    @ns.expect(cnn_model_parameters)
    def post(self):
        """Create a CNN model with the given parameters."""
        logger.info("Creating CNN model with parameters")
        data = request.get_json()
        
        if 'model_id' not in data or not 'uuid' in data:
            return bad_request_message("model_id and uuid are required")
        
        # Convert model_id to string to match keys in possible_models
        model_id = str(data['model_id'])
        logger.info(f"Creating CNN model with parameters: {model_id}")
        possible_models = get_possible_models(data['uuid'])
        
        if model_id not in possible_models:
            return bad_request_message()
        
        json_to_send = possible_models[model_id]
        # URL of the server to which parameters will be sent
        server_url = "http://127.0.0.1:5000/create_cnn_model"
        
        # Make the POST request to the other server with the parameters
        try:
            response = requests.post(server_url, json=json_to_send)
            
            # Check if the response was successful
            if response.status_code == 200:
                return ok_message(response.json())
            else:
                return response_message(response.json(), response.status_code)

        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return runtime_error_message()

@ns.route('/create_child')
class CreateChild(Resource):
    @ns.expect(child_model_parameters)
    def post(self):
        """Create a CNN model with the given parameters."""
        logger.info("Creating CNN model with parameters")
        data = request.get_json()
        
        if 'model_id' not in data or 'second_model_id' not in data:
            return bad_request_message("model_id and second_model_id are required")
        
        model_id = str(data['model_id'])
        second_model_id = str(data['second_model_id'])
        json_to_send = {
            "1": possible_models[model_id],
            "2": possible_models[second_model_id] 
        }
        # URL of the server to which parameters will be sent
        server_url = "http://127.0.0.1:5002/create_child"
        
        # Make the POST request to the other server with the parameters
        try:
            response = requests.post(server_url, json=json_to_send)
            
            # Check if the response was successful
            if response.status_code == 200:
                return ok_message(response.json())
            else:
                return response_message(response.json(), response.status_code)

        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return runtime_error_message()


@ns.route('/create_initial_poblation')
class CreateInitialPoblation(Resource):
    @ns.expect(initial_poblation)
    def post(self):
        """Create a CNN model with the given parameters."""
        logger.info("Creating CNN model with parameters")
        data = request.get_json()
        
        if not check_initial_poblation(data):
            return bad_model_message("Invalid intial poblation request, check response params")
        
        json_to_send = data
        
        server_url = "http://127.0.0.1:5002/create_initial_poblation"
        
        # Make the POST request to the other server with the parameters
        try:
            response = requests.post(server_url, json=json_to_send)
            
            # Check if the response was successful
            if response.status_code == 200:
                json_data = response.json()
                message = json_data.get('message', {})
                models = message.get('models', {})
                models__uuid = generate_uuid()
                # save the models to a file in ./models+uuid.json 
                path = os.path.join(os.path.dirname(__file__),'models', f'{models__uuid}.json')
                with open(path, 'w') as file:
                    json.dump(models, file)
                
                
                return_message = {
                    "uuid": models__uuid,
                    "path": path
                }
                return ok_message(return_message)
            else:
                return response_message(response.json(), response.status_code)

        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return runtime_error_message()

