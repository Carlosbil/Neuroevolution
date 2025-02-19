from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import colorlog
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

# Configure global logger only once.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Avoid duplicate log messages: clear any existing handlers and disable propagation.
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

# Create a colored log handler for console output.
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

# Mapping for activation functions and optimizers.
MAP_ACTIVATE_FUNCTIONS: Dict[str, type] = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'selu': nn.SELU
}

MAP_OPTIMIZERS: Dict[str, type] = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}


def get_individual(params: Dict) -> Dict:
    """
    Define the individual (architecture) based on input parameters.
    
    :param params: Input parameters for the architecture.
    :return: Dictionary representing the individual architecture.
    """
    individual = {
        'num_conv_layers': params['Number of Convolutional Layers'],
        'fully_connected': params['Number of Fully Connected Layers'],
        'filters': params['filters'],
        'kernel_sizes': params['kernel_sizes'],
        'dropout': params['Dropout Rate'],
        'activation': params['Activation Functions'],
    }
    logger.debug(f"Individual created: {individual}")
    return individual


def get_dataset_params(params: Dict) -> Dict:
    """
    Define the dataset parameters based on input parameters.
    
    :param params: Input parameters for the dataset.
    :return: Dictionary containing dataset parameters.
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
    logger.debug(f"Dataset parameters: {dataset_params}")
    return dataset_params


def build_conv_layers(individual: Dict, num_channels: int, px_h: int, px_w: int) -> Tuple[List[nn.Module], int]:
    """
    Build convolutional blocks and return the list of layers and the output size.
    
    :param individual: Architecture parameters.
    :param num_channels: Number of input channels.
    :param px_h: Height of the input image.
    :param px_w: Width of the input image.
    :return: A tuple containing the list of convolutional layers and the flattened output size.
    """
    layers: List[nn.Module] = []
    num_layers = individual['num_conv_layers']
    activations = individual['activation']
    # List of activation functions.
    activation_functions = [MAP_ACTIVATE_FUNCTIONS[act] for act in activations]
    current_channels = num_channels

    for i in range(num_layers):
        out_channels = individual['filters'][i]
        kernel_size = individual['kernel_sizes'][i]

        # Convolutional layer.
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, padding=1))
        
        # Apply BatchNorm if applicable.
        if current_channels > 1 or i > 0:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function (rotate through list if there are more layers than functions).
        layers.append(activation_functions[i % len(activation_functions)]())
        
        # Down-sampling: the last layer uses stride 1 to preserve more information.
        if i < num_layers - 1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        
        current_channels = out_channels

    # Compute the output size after the convolutional layers.
    temp_model = nn.Sequential(*layers)
    dummy_input = torch.zeros(1, num_channels, px_h, px_w)
    output_size = temp_model(dummy_input).view(-1).shape[0]
    return layers, output_size


def build_fc_layers(output_size: int, num_fc: int, dropout: float, num_classes: int) -> List[nn.Module]:
    """
    Build fully connected layers and return the list of layers.
    
    :param output_size: Size of the flattened convolutional output.
    :param num_fc: Number of fully connected layers.
    :param dropout: Dropout rate.
    :param num_classes: Number of output classes.
    :return: List of fully connected layers.
    """
    fc_layers: List[nn.Module] = []
    for _ in range(num_fc):
        fc_layers.append(nn.Linear(in_features=output_size, out_features=output_size))
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
            dropout -= 1  # Reduce dropout for each layer.
    fc_layers.append(nn.Linear(output_size, num_classes))
    return fc_layers


def train_and_evaluate(model: nn.Module, device: torch.device,
                       train_loader: DataLoader, test_loader: DataLoader,
                       optimizer: optim.Optimizer, criterion: nn.Module,
                       num_epochs: int = 3) -> float:
    """
    Perform the training loop and, after each epoch, evaluate the model.
    Uses tqdm to show progress for each batch.
    
    :param model: The CNN model.
    :param device: Device to run training on.
    :param train_loader: Training data loader.
    :param test_loader: Testing data loader.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :param num_epochs: Number of training epochs.
    :return: The final evaluation accuracy (percentage).
    """
    final_test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training.
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        logger.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Evaluation after each epoch.
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch} Evaluation", unit="batch")
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                test_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / total
        logger.info(f"Epoch {epoch} Evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        final_test_acc = test_acc

    return final_test_acc


def build_cnn_from_individual(individual: Dict, num_channels: int, px_h: int, px_w: int, num_classes: int,
                              train_loader: DataLoader, test_loader: DataLoader,
                              optimizer_name: str, learning_rate: float, num_epochs: int = 3) -> float:
    """
    Build the CNN model based on the individual, then train and evaluate it using a combined training and evaluation loop.
    
    :param individual: Architecture parameters.
    :param num_channels: Number of input channels.
    :param px_h: Height of the input image.
    :param px_w: Width of the input image.
    :param num_classes: Number of output classes.
    :param train_loader: Training data loader.
    :param test_loader: Testing data loader.
    :param optimizer_name: Name of the optimizer.
    :param learning_rate: Learning rate.
    :param num_epochs: Number of training epochs.
    :return: The final evaluation accuracy (percentage).
    """
    # Build convolutional blocks and compute the output size.
    conv_layers, conv_output_size = build_conv_layers(individual, num_channels, px_h, px_w)
    
    # Build fully connected layers.
    fc_layers = build_fc_layers(conv_output_size, individual['fully_connected'], individual['dropout'], num_classes)
    
    # Assemble the complete model.
    layers = conv_layers + [nn.Flatten()] + fc_layers
    model = nn.Sequential(*layers)
    logger.debug(f"Constructed CNN model:\n{model}")
    
    # Set up device, optimizer, and loss function.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    optimizer_class = MAP_OPTIMIZERS[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate using the combined loop.
    final_accuracy = train_and_evaluate(model, device, train_loader, test_loader, optimizer, criterion, num_epochs)
    
    return final_accuracy
