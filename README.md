# Neuroevolution System

## Overview

Neuroevolution is a distributed system for evolving neural network architectures using genetic algorithms. The system uses Kafka for communication between its components, allowing for scalable and fault-tolerant evolution of neural network models.

## Architecture

The system consists of three main components:

### 1. Broker

The Broker component serves as the central communication hub, managing the flow of messages between different parts of the system. It handles:

- Creating initial populations of neural network architectures
- Evaluating populations
- Selecting the best architectures
- Creating child models through genetic operations
- Orchestrating the entire genetic algorithm workflow

### 2. Genome

The Genome component is responsible for the genetic operations on neural network architectures:

- Creating initial populations of neural network models
- Crossing genomes (combining two parent architectures)
- Mutating genomes to introduce variation

### 3. Evolutioners

The Evolutioners component handles the creation and evaluation of CNN models:

- Creating CNN models from genome specifications
- Training and evaluating models
- Reporting performance metrics back to the system

## Communication Flow

The components communicate through Kafka topics:

```
Broker → Genome → Evolutioners → Broker
```

## Installation

### Prerequisites

- Python 3.8+
- Kafka server running (default: localhost:9092)
- Docker and Docker Compose (optional, for containerized deployment)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/neuroevolution.git
cd neuroevolution
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Kafka (if not using Docker):

Ensure Kafka is running at `localhost:9092` or update the `KAFKA_BROKER` variable in the configuration files.

4. Using Docker (optional):

```bash
docker-compose up -d
```

## Usage

### Starting the Services

Start each component in a separate terminal:

```bash
# Terminal 1: Start the Broker
cd Broker
python app.py

# Terminal 2: Start the Genome service
cd Genome
python app.py

# Terminal 3: Start the Evolutioners service
cd Evolutioners
python app.py
```

### Running a Genetic Algorithm

To start the genetic algorithm process, send a message to the `genetic-algorithm` topic with the following parameters:

```json
{
  "num_channels": 1,
  "px_h": 28,
  "px_w": 28,
  "num_classes": 10,
  "batch_size": 32,
  "num_poblation": 10
}
```

This will:
1. Create an initial population of neural network architectures
2. Evaluate each architecture
3. Select the best performing architectures
4. Create new child architectures through crossover and mutation

## API Reference

### Kafka Topics

#### Input Topics

- `create-initial-population`: Create initial population of neural networks
- `evaluate-population`: Evaluate a population of neural networks
- `select-best-architectures`: Select the best architectures from a population
- `create-child`: Create a child model from two parent models
- `genetic-algorithm`: Run the complete genetic algorithm workflow

#### Response Topics

Each input topic has a corresponding response topic with the suffix `-response`.

### Message Formats

#### Create Initial Population

```json
{
  "num_channels": 1,
  "px_h": 28,
  "px_w": 28,
  "num_classes": 10,
  "batch_size": 32,
  "num_poblation": 10
}
```

#### Evaluate Population

```json
{
  "uuid": "population-uuid"
}
```

#### Create Child

```json
{
  "model_id": "parent1-id",
  "second_model_id": "parent2-id",
  "uuid": "population-uuid"
}
```

## Testing

The project includes integration tests to verify the functionality of each component:

```bash
# Run Broker integration tests
cd Broker
python integration_tests.py

# Run Evolutioners integration tests
cd Evolutioners
python integration_tests.py

# Run Genome integration tests
cd Genome
python integration_tests.py
```

## License

This project is licensed under the terms of the license included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.