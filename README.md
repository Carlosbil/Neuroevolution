 # NeuroEvolution Distributed System

This project is a distributed system for training neural networks using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. The system is designed to be scalable and fault-tolerant, and can be used to train neural networks on large datasets.

## Features

- Distributed training of neural networks using microservices architecture
- Fault-tolerant design with automatic recovery from failures
- Scalable architecture that can be easily scaled up or down based on workload
- Support for training neural networks on large datasets
- All the microservices has a REST API that can be used to interact with the system
- All the microservices are implemented in Python using the Flask framework
- All the microservices has a Swagger to interact with the API and test the endpoints

## Architecture

The system is composed of several microservices that work together to train neural networks using the NEAT algorithm. The microservices are:

- **Broker**: create poblation, distribute genomes and evaluate them. Save the poblation. `Port: 5001`
- **Genome**: Responsible for create a poblation and new childs. `Port: 5002`
- **Evolutioners**: Responsible for training and evauluating genomes. `Port: 5000`

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

## BROKER
---
The broker just call the evolutioners and genome endpoints in order to complete the NEAT algorythm. 

1. First will call to the genome endpoint in order to create the initial poblation 
2. Then will call the evolutioners endpoint in order to train the genomes and save their scores. 
3. Select the best genomes and create a new poblation
4. Repeat the process until the poblation is trained

### Endpoints

- **/create_initial_poblation**: Create the initial poblation and save into `/models/uuid.json`
- **/create_cnn_model**: Create train and evaluate a CNN from a genome of a `uuid` poblation
- **/create_child**: Create a new child from 2 genomes of a `uuid` poblation

## GENOME
---
This microservice manage the genomes to create new childs and create the initial poblation

### Endpoints
- **/create_child**: Create a new child from 2 genomes jsons
- **/create_initial_poblation**: Create a random poblation

## EVOLUTIONERS
---
This microservice create train and evaluate the genomes returning their scores

### Endpoints
- **/create_cnn_model**: Train a genome and return the score