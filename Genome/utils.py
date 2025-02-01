import torch
import torch.nn as nn
import torch.optim as optim
import logging
import colorlog
import random
logging.basicConfig(level=logging.DEBUG,  # Mínimo nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato del log
                    handlers=[logging.StreamHandler()])  # Mostrar los logs en consola
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Crear un handler para los logs en consola con colores
log_handler = colorlog.StreamHandler()

# Define el formato de los logs con colores
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',      # Debug será de color cian
        'INFO': 'green',      # Info será de color verde
        'WARNING': 'yellow',  # Warning será de color amarillo
        'ERROR': 'red',       # Error será de color rojo
        'CRITICAL': 'bold_red'  # Critical será de color rojo negrita
    }
)

# Establecer el formatter en el handler
log_handler.setFormatter(formatter)

# Añadir el handler al logger
logger.addHandler(log_handler)

def get_individual(params):
    """ 
    Define the individual based on the parameters.
    """
    conv_layers = params['Number of Convolutional Layers']
    fc_layers = params['Number of Fully Connected Layers']
    nodes = params['Number of Nodes in Each Layer']
    activations = params['Activation Functions']
    dropout_rate = params['Dropout Rate']
    learning_rate = params['Learning Rate']
    batch_size = params['Batch Size']
    optimizer_name = params['Optimizer'].lower()
    kernel_sizes = params['kernel_sizes']

    num_channels = params['num_channels']
    px_h = params['px_h']
    px_w = params['px_w']
    num_classes = params['num_classes']
    filters = params['filters']
    
    individual = {
        'num_conv_layers': conv_layers,
        'fully_connected': fc_layers,
        'filters': filters,
        'kernel_sizes': kernel_sizes,
        'dropout': dropout_rate,
        'activation': activations,
    }
    return individual

def get_dataset_params(params):
    """ 
    Define the dataset parameters based on the input.
    """
    dataset_params = {
        'num_channels': params['num_channels'],
        'px_h': params['px_h'],
        'px_w': params['px_w'],
        'num_classes': params['num_classes'],
        'batch_size': params['Batch Size'],
        'optimizer': params['Optimizer'].lower(),
        'learning_rate': params['Learning Rate'],
    }

    return dataset_params


def create_genome(individual, dataset_params):
    """ 
    Function to create a genome based on the individual and dataset parameters.
    """
    genome = {
        'Number of Convolutional Layers': individual['num_conv_layers'],
        'Number of Fully Connected Layers': individual['fully_connected'],
        'Number of Nodes in Each Layer': [64] * individual['fully_connected'],
        'Activation Functions': [0] * individual['fully_connected'],
        'Dropout Rate': individual['dropout'],
        'Learning Rate': dataset_params['learning_rate'],
        'Batch Size': dataset_params['batch_size'],
        'Optimizer': MAP_OPTIMIZERS[dataset_params['optimizer']],
        'num_channels': dataset_params['num_channels'],
        'px_h': dataset_params['px_h'],
        'px_w': dataset_params['px_w'],
        'num_classes': dataset_params['num_classes'],
        'filters': individual['filters'],
        'kernel_sizes': individual['kernel_sizes']
    }
    return genome

def cross_genomes(genome1, genome2):
    """ 
    Function to cross two genomes. 50% of the time, the gene will be taken from genome1, otherwise from genome2. Randomly.
    """
    new_genome = {}
    for key in genome1:
        new_genome[key] = genome1[key] if torch.rand(1) > 0.5 else genome2[key]
        
    return new_genome


def mutate_genome(genome, mutation_rate=0.1):
    """ 
    Function to mutate a genome. Randomly changes a gene with a given mutation rate.Probability of mutation is given by mutation_rate.
    Currently, the mutation rate is set to 0.1. (10% chance of mutation)
    """
    for key in genome:
        if torch.rand(1) < mutation_rate:
            logger.debug(f"Mutating key: {key} with value: {genome[key]}")
            
            if key in ['Number of Convolutional Layers', 'Number of Fully Connected Layers']:
                genome[key] = torch.randint(1, 5, (1,)).item()
            
            elif key == 'Number of Nodes in Each Layer':
                genome[key] = [torch.randint(16, 128, (1,)).item() for _ in range(genome['Number of Fully Connected Layers'])]
            
            elif key == 'Activation Functions':
                genome[key] = [torch.randint(0, len(MAP_ACTIVATE_FUNCTIONS), (1,)).item() for _ in range(genome['Number of Fully Connected Layers'])]
            
            elif key in ['Dropout Rate', 'Learning Rate']:
                factor = random.choice([0.1, 0.01, 0.001])
                genome[key] *= factor if random.random() < 0.5 else 1 / factor
                genome[key] = min(round(genome[key], 4), 0.9)  # Asegurar que no supere 1
            
            elif key == 'Batch Size':
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'Optimizer':
                genome[key] = torch.randint(0, len(MAP_OPTIMIZERS), (1,)).item()
            
            elif key in ['num_channels', 'num_classes']:
                genome[key] = torch.randint(1, 4, (1,)).item() if key == 'num_channels' else torch.randint(2, 10, (1,)).item()
            
            elif key in ['px_h', 'px_w']:
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'filters':
                genome[key] = [torch.randint(1, 5, (1,)).item() for _ in range(genome['Number of Convolutional Layers'])]
            
            elif key == 'kernel_sizes':
                genome[key] = [torch.randint(3, 7, (1,)).item() for _ in range(genome['Number of Convolutional Layers'])]
            
            logger.debug(f"New value for {key}: {genome[key]}")
    
    return genome


MAP_ACTIVATE_FUNCTIONS = {
    
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'selu': nn.SELU
}

MAP_OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}
