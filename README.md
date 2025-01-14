 # NeuroEvolution Distributed System

This project is a distributed system for training neural networks using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. The system is designed to be scalable and fault-tolerant, and can be used to train neural networks on large datasets.

## Features

- Distributed training of neural networks using microservices architecture
- Fault-tolerant design with automatic recovery from failures
- Scalable architecture that can be easily scaled up or down based on workload
- Support for training neural networks on large datasets

## Architecture

The system is composed of several microservices that work together to train neural networks using the NEAT algorithm. The microservices are:
- **Broker**: Responsible for distributing work to workers and collecting results
- **Genome Store**: Responsible for storing and retrieving genomes
- **Evolutioners**: Responsible for training neural networks using the NEAT algorithm

## Possible Genomes

The system supports training neural networks with different types of genomes, including:
- **Number of Convolutional Layers**: The number of convolutional layers in the network
- **Number of Fully Connected Layers**: The number of fully connected layers in the network
- **Number of Nodes in Each Layer**: The number of nodes in each layer of the network
- **Activation Functions**: The activation functions used in the network
- **Dropout Rate**: The dropout rate used in the network
- **Learning Rate**: The learning rate used in the network
- **Batch Size**: The batch size used in the network
- **Optimizer**: The optimizer used in the network