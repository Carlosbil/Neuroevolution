# endpoints.py
from flask_restx import Resource
from flask import request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from swagger_models import api, cnn_model_parameters
from responses import ok_message, bad_model_message, bad_optimizer_message
from utils import get_individual, get_dataset_params, build_cnn_from_individual, MAP_OPTIMIZERS, logger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
ns = api.namespace('', description='CNN Model operations')


@ns.route('/create_cnn_model')
class CNNModel(Resource):
    @ns.expect(cnn_model_parameters)
    def post(self):
        """Create, train and evaluate a CNN model with the given parameters."""
        params = request.get_json()
        
        logger.info(f"Creating CNN model with parameters: {params}")
        
        try:
            individual = get_individual(params)
            dataset_params = get_dataset_params(params)
            
            # Preparar dataset MNIST y sus dataloaders
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=dataset_params.get('batch_size'), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=dataset_params.get('batch_size'), shuffle=False)
            
            # Construir, entrenar y evaluar el modelo
            accuracy = build_cnn_from_individual(
                individual,
                num_channels=dataset_params.get('num_channels'),
                px_h=dataset_params.get('px_h'),
                px_w=dataset_params.get('px_w'),
                num_classes=dataset_params.get('num_classes'),
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer_name=dataset_params.get('optimizer'),
                learning_rate=dataset_params.get('learning_rate'),
                num_epochs=3
            )
            logger.info(f"Individual created and model trained: {individual}")
        except ValueError as e:
            return bad_model_message()
        
        response = {
            'message': 'CNN model created, trained and evaluated successfully',
            'score': accuracy,
        }
        return ok_message(response)