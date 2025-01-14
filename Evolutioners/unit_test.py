import unittest
import torch.nn as nn
from utils import get_individual, get_dataset_params, build_cnn_from_individual, MAP_OPTIMIZERS, MAP_ACTIVATE_FUNCTIONS

class TestCNNModelFunctions(unittest.TestCase):

    def test_get_individual(self):
        params = {
            'Number of Convolutional Layers': 3,
            'Number of Fully Connected Layers': 2,
            'Number of Nodes in Each Layer': [64, 128],
            'Activation Functions': ['relu', 'sigmoid'],
            'Dropout Rate': 0.5,
            'Learning Rate': 0.001,
            'Batch Size': 32,
            'Optimizer': 'adam',
            'kernel_sizes': [3, 3, 3],
            'num_channels': 3,
            'px_h': 64,
            'px_w': 64,
            'num_classes': 10,
            'filters': [32, 64, 128]
        }

        individual = get_individual(params)
        
        # Test if individual contains the expected keys
        self.assertIn('num_conv_layers', individual)
        self.assertIn('fully_connected', individual)
        self.assertIn('filters', individual)
        self.assertIn('kernel_sizes', individual)
        self.assertIn('dropout', individual)

    def test_missing_key_in_get_individual(self):
        params = {'Number of Convolutional Layers': 3}  # Missing required keys
        with self.assertRaises(KeyError):
            get_individual(params)

    def test_get_dataset_params(self):
        params = {
            'num_channels': 3,
            'px_h': 64,
            'px_w': 64,
            'num_classes': 10,
            'Batch Size': 32,
            'Optimizer': 'sgd',
            'Learning Rate': 0.01
        }

        dataset_params = get_dataset_params(params)

        # Test if dataset_params contains the expected keys
        self.assertIn('num_channels', dataset_params)
        self.assertIn('px_h', dataset_params)
        self.assertIn('px_w', dataset_params)
        self.assertIn('num_classes', dataset_params)
        self.assertIn('batch_size', dataset_params)
        self.assertIn('optimizer', dataset_params)
        self.assertIn('learning_rate', dataset_params)

    def test_missing_key_in_get_dataset_params(self):
        params = {'num_channels': 3, 'px_h': 64}  # Missing required keys
        with self.assertRaises(KeyError):
            get_dataset_params(params)

    def test_invalid_activation_function(self):
        individual = {
            'num_conv_layers': 2,
            'fully_connected': 2,
            'filters': [32, 64],
            'kernel_sizes': [3, 3],
            'dropout': 0.5,
            'activation': ['nonexistent_activation']
        }
        num_channels = 3
        px_h = 64
        px_w = 64
        num_classes = 10

        with self.assertRaises(KeyError):
            build_cnn_from_individual(individual, num_channels, px_h, px_w, num_classes)

    def test_map_activation_functions(self):
        # Test if each activation function is mapped correctly
        for key, func in MAP_ACTIVATE_FUNCTIONS.items():
            self.assertIn(key, MAP_ACTIVATE_FUNCTIONS)
            self.assertTrue(callable(func))

    def test_invalid_activation_key(self):
        with self.assertRaises(KeyError):
            MAP_ACTIVATE_FUNCTIONS['invalid_key']
            
if __name__ == '__main__':
    unittest.main()