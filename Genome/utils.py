import torch
import torch.nn as nn
import torch.optim as optim
import logging
import colorlog
import random
from confluent_kafka import Producer, Consumer
import signal
import sys
import time
KAFKA_BROKER = "localhost:9092"


# Configuración global del logger, incluyendo el nombre del archivo y la línea
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Evitar mensajes duplicados: limpiar cualquier manejador existente y deshabilitar la propagación.
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

# Crear un manejador de log con colores para la salida en consola.
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)
log_handler.setFormatter(formatter)
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
    logger.debug(f"Creating genome with individual: {individual} and dataset_params: {dataset_params}")
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
    logger.debug(f"Crossing genomes: {genome1} and {genome2}")
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
                # Selección aleatoria de funciones de activación
                activation_function_key = random.choice(list(MAP_ACTIVATE_FUNCTIONS.keys()))  # Selección aleatoria de la clave
                genome[key] = [activation_function_key for _ in range(genome['Number of Fully Connected Layers'])]  # Asigna las funciones de activación
            
            elif key in ['Dropout Rate', 'Learning Rate']:
                factor = random.choice([0.1, 0.01, 0.001])
                genome[key] *= factor if random.random() < 0.5 else 1 / factor
                genome[key] = min(round(genome[key], 4), 0.9)  # Asegurar que no supere 1
            
            elif key == 'Batch Size':
                genome[key] = torch.randint(16, 128, (1,)).item()
            
            elif key == 'Optimizer':
                # Selección aleatoria de un optimizador del mapa
                optimizer_key = random.choice(list(MAP_OPTIMIZERS.keys()))  # Selección aleatoria de la clave
                genome[key] = optimizer_key  # Asigna el nombre del optimizador
            
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


def check_initial_poblation(params):
    """ 
    Check the initial poblation parameters.
    """
    if 'num_channels' not in params or 'px_h' not in params or 'px_w' not in params or 'num_classes' not in params or 'batch_size' not in params or 'num_poblation' not in params:
        return False
    return True


def generate_random_model_config(num_channels: int, px_h: int, px_w: int, num_classes: int, batch_size: int) -> dict:
    """
    Generate a CNN model configuration with fixed parameters for input channels, image dimensions, 
    number of classes, and batch size, while the rest of the parameters are generated randomly 
    with sensible values for an artificial neural network.
    
    :param num_channels: Number of input channels (fixed).
    :param px_h: Image height (fixed).
    :param px_w: Image width (fixed).
    :param num_classes: Number of output classes (fixed).
    :param batch_size: Batch size (fixed).
    :return: A dictionary representing the CNN model configuration.
    """
    # Randomly determine the number of convolutional and fully connected layers.
    num_conv_layers = random.randint(1, 4)
    num_fc_layers = random.randint(1, 3)
    
    # Generate a list for "Number of Nodes in Each Layer" with random sensible node counts.
    nodes_options = [16, 32, 64, 128]
    fc_nodes = [random.choice(nodes_options) for _ in range(num_fc_layers)]
    
    # Generate a list for "Activation Functions" by randomly choosing from common activations.
    activation_options = ["relu", "sigmoid", "tanh", "leakyrelu", "selu"]
    activations = [random.choice(activation_options) for _ in range(num_fc_layers)]
    
    # Generate a random dropout rate between 0.1 and 0.5 (rounded to 2 decimals).
    dropout_rate = round(random.uniform(0.1, 0.5), 2)
    
    # Choose a random learning rate from a predefined set.
    learning_rate_options = [0.1, 0.01, 0.001, 0.0001]
    learning_rate = random.choice(learning_rate_options)
    
    # Randomly select an optimizer from a list of common optimizers.
    optimizer_options = ["adam", "adamw", "sgd", "rmsprop"]
    optimizer = random.choice(optimizer_options)
    
    # Generate a list for "filters" for each convolutional layer (common choices for number of filters).
    filter_options = [16, 32, 64]
    filters = [random.choice(filter_options) for _ in range(num_conv_layers)]
    
    # Generate a list for "kernel_sizes" for each convolutional layer.
    kernel_options = [3, 5, 7]
    kernel_sizes = [random.choice(kernel_options) for _ in range(num_conv_layers)]
    
    # Assemble the model configuration dictionary.
    model_config = {
        "Number of Convolutional Layers": num_conv_layers,
        "Number of Fully Connected Layers": num_fc_layers,
        "Number of Nodes in Each Layer": fc_nodes,
        "Activation Functions": activations,
        "Dropout Rate": dropout_rate,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Optimizer": optimizer,
        "num_channels": num_channels,
        "px_h": px_h,
        "px_w": px_w,
        "num_classes": num_classes,
        "filters": filters,
        "kernel_sizes": kernel_sizes
    }
    
    return model_config


def produce_message(producer, topic, message, times=10):
    """Publish a message to Kafka."""
    for i in range(times):
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        print(f"[✅] Mensaje enviado: {message} - Tópico: {topic} - {i} ")

def create_producer():
    """Crea un productor de Kafka."""
    return Producer({
        'bootstrap.servers': KAFKA_BROKER,
        'linger.ms': 0,  # clave para envío inmediato
        'batch.size': 1, # opcional
        })

def create_consumer():
    """Crea y configura un consumidor de Kafka con mejor manejo de errores."""
    return Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'genome-consumer-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,  # Asegura que los offsets se confirmen automáticamente
        'session.timeout.ms': 60000,  # Evita desconexiones prematuras
        'heartbeat.interval.ms': 15000,  # Reduce el riesgo de expulsión por latencia
        'max.poll.interval.ms': 300000,  # Permite más tiempo para procesar mensajes
    })

MAP_ACTIVATE_FUNCTIONS = {
    'relu': 'relu',
    'sigmoid':'sigmoid',
    'tanh': 'tanh',
    'leakyrelu': 'leakyrelu',
}

MAP_OPTIMIZERS = {
    'adam': 'adam',
    'adamw': 'adamw',
    'sgd': 'sgd',
    'rmsprop': 'rmsprop',
}
