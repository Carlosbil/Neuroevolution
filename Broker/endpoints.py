# endpoints.py
from flask_restx import Resource
from flask import request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger
from swagger_models import api, cnn_model_parameters
from responses import ok_message, bad_model_message, bad_optimizer_message, runtime_error_message, response_message
import requests

ns = api.namespace('', description='CNN Model operations')


@ns.route('/create_cnn_model')
class CNNModel(Resource):
    @ns.expect(cnn_model_parameters)
    def post(self):
        """Create a CNN model with the given parameters."""
        logger.info(f"Creating CNN model with parameters")
        params = request.get_json()
        
        json_to_send = {
                "Number of Convolutional Layers": 2,
                "Number of Fully Connected Layers": 2,
                "Number of Nodes in Each Layer": [
                    32
                ],
                "Activation Functions": [
                    "relu"
                ],
                "Dropout Rate": 0.2,
                "Learning Rate": 0.0001,
                "Batch Size": 32,
                "Optimizer": "adamw",
                "num_channels": 3,
                "px_h": 32,
                "px_w": 32,
                "num_classes": 2,
                "filters": [
                    3,3
                ],
                "kernel_sizes": [
                    3,3
                ]
            }
        # URL del servidor al que se enviarán los parámetros
        server_url = "http://127.0.0.1:5000/create_cnn_model"
        
        # Realizar la solicitud POST con los parámetros al otro servidor
        try:
            response = requests.post(server_url, json=json_to_send)
            
            # Verifica que la respuesta fue exitosa
            if response.status_code == 200:
                return ok_message(response.json())
            else:
                return response_message(response.json(), response.status_code)

        except Exception as e:
            return runtime_error_message()