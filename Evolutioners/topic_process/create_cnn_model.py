from utils import logger, get_dataset_params, get_individual, build_cnn_from_individual, check_params
from responses import ok_message, bad_model_message
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def handle_create_cnn_model(topic, params):
    """
    Train and evaliate a cnn model with the given parameters.
    """
    logger.info(f"Processing cnn model: {params}")
    try:
        if not check_params(params):
            return bad_model_message(topic)
        
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

        response = {
            'message': 'CNN model created, trained and evaluated successfully',
            'score': accuracy,
        }
        return ok_message(topic, response)
    except ValueError as e:
        return bad_model_message(topic)
    
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return bad_model_message(topic)