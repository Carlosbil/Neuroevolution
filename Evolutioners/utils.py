import torch
import torch.nn as nn
import torch.optim as optim
import logging
import colorlog
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


def build_cnn_from_individual(individual, num_channels, px_h, px_w, num_classes):
    """ 
    Function to build a CNN model based on a dictionary of parameters.
    """
    layers = []
    num_layers = individual['num_conv_layers']
    fully_connected = individual['fully_connected']
    dropout = individual['dropout']
    activations = individual['activation']
    
    out_channels_previous_layer = num_channels
    activation_functions = [MAP_ACTIVATE_FUNCTIONS[act] for act in activations]

    for i in range(num_layers):
        out_channels = individual['filters'][i]
        kernel_size = individual['kernel_sizes'][i]
        
        conv_layer = nn.Conv2d(out_channels_previous_layer, out_channels, kernel_size=kernel_size, padding=1)
        layers.append(conv_layer)
        
        if out_channels_previous_layer > 1 or i > 0:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Use activation function from the list
        layers.append(activation_functions[i % len(activation_functions)]())
        
        if i < num_layers - 1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Use a stride of 2 for down-sampling
        else:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=1))  # Final layer might have stride of 1 to preserve size

        out_channels_previous_layer = out_channels

    # Temporarily create the model to calculate the output size from convolution layers
    temp_model = nn.Sequential(*layers)

    # Create a dummy tensor to calculate the output size
    dummy_input = torch.zeros(1, num_channels, px_h, px_w)
    output_size = temp_model(dummy_input).view(-1).shape[0]

    layers.append(nn.Flatten())
    
    # Adding fully connected layers
    for i in range(fully_connected):
        layers.append(nn.Linear(in_features=output_size, out_features=output_size))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            dropout -= 1  # Decrease the dropout for each layer

    layers.append(nn.Linear(output_size, num_classes))
    return nn.Sequential(*layers)


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
