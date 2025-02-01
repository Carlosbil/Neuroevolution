import unittest
import torch.nn as nn
from utils import mutate_genome, create_genome, cross_genomes, MAP_OPTIMIZERS, MAP_ACTIVATE_FUNCTIONS


class TestGenomeFunctions(unittest.TestCase):
    def setUp(self):
        self.genome = {
            'Dropout Rate': 0.5,
            'Learning Rate': 0.01,
            'Number of Fully Connected Layers': 3
        }
        self.mutation_rate = 1.0
        self.logger = type('Logger', (), {"debug": print})
    
    def test_mutate_genome(self):
        mutated = mutate_genome(self.genome.copy(), self.mutation_rate)
        self.assertLessEqual(mutated['Dropout Rate'], 1.0)
        self.assertLessEqual(mutated['Learning Rate'], 1.0)
        self.assertIn(mutated['Number of Fully Connected Layers'], range(1, 5))
    
    def test_create_genome(self):
        individual = {'num_conv_layers': 2, 'fully_connected': 3, 'dropout': 0.2, 'filters': [32, 64], 'kernel_sizes': [3, 5]}
        dataset_params = {'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'num_channels': 3, 'px_h': 128, 'px_w': 128, 'num_classes': 10}
        genome = create_genome(individual, dataset_params)
        self.assertEqual(genome['Number of Convolutional Layers'], 2)
        self.assertEqual(genome['Learning Rate'], 0.001)
        self.assertEqual(len(genome['filters']), 2)
    
    def test_cross_genomes(self):
        genome1 = {'param1': 1, 'param2': 2, 'param3': 3}
        genome2 = {'param1': 4, 'param2': 5, 'param3': 6}
        crossed = cross_genomes(genome1, genome2)
        self.assertIn(crossed['param1'], [1, 4])
        self.assertIn(crossed['param2'], [2, 5])
        self.assertIn(crossed['param3'], [3, 6])
        
    def test_cross_genomes_different_values(self):
        genome1 = {'a': 10, 'b': 20, 'c': 30}
        genome2 = {'a': 40, 'b': 50, 'c': 60}
        crossed = cross_genomes(genome1, genome2)
        for key in genome1:
            self.assertIn(crossed[key], [genome1[key], genome2[key]])

if __name__ == "__main__":
    unittest.main()