# endpoints.py
from flask_restx import Resource
from flask import request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from swagger_models import api, cnn_model_parameters, initial_poblation
from responses import ok_message, bad_model_message, bad_optimizer_message
from utils import get_individual, get_dataset_params, MAP_OPTIMIZERS, logger, create_genome, \
    cross_genomes, mutate_genome, check_initial_poblation, generate_random_model_config

ns = api.namespace('', description='Genetic Algorithm operations')


@ns.route('/create_child')
class CNNModel(Resource):
    @ns.expect(cnn_model_parameters)
    def post(self):
        """Create a CNN model with the given parameters."""
        params = request.get_json()
        
        logger.info(f"Creating CNN model with parameters: {params}")
        
        try:
            individual = get_individual(params['1'])
            second_individual = get_individual(params['2'])
            dataset_params = get_dataset_params(params['1'])
            second_dataset_params = get_dataset_params(params['2'])
            
            gen_1 = create_genome(individual, dataset_params)
            gen_2 = create_genome(second_individual, second_dataset_params)
            
            crossed_individual = cross_genomes(genome1=gen_1, genome2=gen_2)
            mutated_individual = mutate_genome(crossed_individual)
            
            logger.info(f"Individual crossed and mutated: gen1={gen_1}, gen2={gen_2}, crossed={crossed_individual}, mutated={mutated_individual}")
        except ValueError as e:
            return bad_model_message()
        
        response = {
            'message': 'CNN model crossed and mutated successfully',
            'model': mutated_individual,
        }
        return ok_message(response)

@ns.route('/create_initial_poblation')
class CNNModel(Resource):
    @ns.expect(initial_poblation)
    def post(self):
        """Create a CNN model with the given parameters."""
        params = request.get_json()
        
        logger.info(f"Creating CNN model with parameters: {params}")
        
        if not check_initial_poblation(params):
            return bad_model_message("Invalid intial poblation request, check response params")
        
        num_poblation = params.get('num_poblation', 10)
        models = {}
        for i in range(num_poblation):
            models[i] = generate_random_model_config(params.get('num_channels', 3),
                                                     params.get('px_h', 32),
                                                     params.get('px_w', 32),
                                                     params.get('num_classes', 10),
                                                     params.get('batch_size', 32)
                                                     )
        
        
        response = {
            'message': 'CNN model crossed and mutated successfully',
            'models': models
        }
        return ok_message(response)
