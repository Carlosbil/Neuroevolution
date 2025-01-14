# endpoints.py
from flask_restx import Resource
from flask import request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from swagger_models import api, cnn_model_parameters
from responses import ok_message, bad_model_message, bad_optimizer_message
from utils import get_individual, get_dataset_params, build_cnn_from_individual, MAP_OPTIMIZERS, logger

ns = api.namespace('cnn', description='CNN Model operations')


@ns.route('/create_cnn_model')
class CNNModel(Resource):
    @ns.expect(cnn_model_parameters)
    def post(self):
        """Create a CNN model with the given parameters."""
        params = request.get_json()
        
        logger.info(f"Creating CNN model with parameters: {params}")
        
        try:
            individual = get_individual(params)
            dataset_params = get_dataset_params(params)
            model = build_cnn_from_individual(individual, dataset_params.get('num_channels'), dataset_params.get('px_h'), dataset_params.get('px_w'), dataset_params.get('num_classes'))
            logger.info(f"Individual created: {individual}")
        except ValueError as e:
            return bad_model_message()

        # Select optimizer
        if dataset_params.get('optimizer') in MAP_OPTIMIZERS:
            optimizer = MAP_OPTIMIZERS[dataset_params.get('optimizer')]
            model.optimizer = optimizer(model.parameters(), lr=dataset_params.get('learning_rate'))
            logger.info(f"Optimizer selected: {dataset_params.get('optimizer')}")
        else:
            return bad_optimizer_message()
        
        response = {
            'message': 'CNN model created successfully',
            'model': str(model),
            'batch_size': dataset_params.get('batch_size'),
        }
        return ok_message(response)
