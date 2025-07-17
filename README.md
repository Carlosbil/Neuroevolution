# Neuroevolution System

## Overview

Neuroevolution is a distributed system for evolving neural network architectures using genetic algorithms. The system uses Kafka for communication between its components, allowing for scalable and fault-tolerant evolution of neural network models.

## Architecture

The system consists of four main components:

### 1. Broker

The Broker component serves as the central communication hub, managing the flow of messages between different parts of the system. It handles:

- Creating initial populations of neural network architectures
- Evaluating populations
- Selecting the best architectures
- Creating child models through genetic operations
- Providing status check endpoints for other services

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

### 4. GeneticAlgorithmService

The GeneticAlgorithmService is a dedicated microservice that orchestrates the complete genetic algorithm workflow:

- Manages multi-generation evolutionary cycles
- Implements convergence criteria and stopping conditions
- Coordinates with the broker for population operations
- Prevents blocking of the main broker during long-running evolutionary processes

## Communication Flow

The components communicate through Kafka topics:

```
Client → Broker → GeneticAlgorithmService → Broker → Genome/Evolutioners
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

### Running a Complete Genetic Algorithm

To start the **complete genetic algorithm** with multiple generations, send a message to the `genetic-algorithm` topic with the following parameters:

```json
{
  "num_channels": 1,
  "px_h": 28,
  "px_w": 28,
  "num_classes": 10,
  "batch_size": 32,
  "num_poblation": 10,
  "max_generations": 20,
  "fitness_threshold": 0.95
}
```

This will execute the full evolutionary process:
1. **Create initial population** of neural network architectures
2. **For each generation**:
   - Evaluate each architecture's performance
   - Select the best performing architectures (50%)
   - Create new child architectures through crossover and mutation
   - Replace the population with the new generation
3. **Check convergence criteria**:
   - Maximum generations reached
   - Fitness threshold achieved
   - Fitness stagnation detected
4. **Return best model** and evolution statistics

### Running Hybrid NEAT

To start the Hybrid NEAT algorithm, send a message to the `start-hybrid-neat` topic:

```json
{
  "num_poblation": 15,
  "max_generations": 30,
  "fitness_threshold": 0.98
}
```

This launches the complete genetic algorithm with NEAT-optimized parameters.

## Hybrid NEAT Philosophy

The system incorporates **Hybrid NEAT** (NeuroEvolution of Augmenting Topologies) principles to maximize each individual's potential through adaptive learning capabilities. The philosophy behind this approach is based on the principle that:

> **"The capacity to adapt and learn is superior to having a good foundation but becoming stagnant"**

### Why Hybrid NEAT?

Traditional genetic algorithms often focus on finding good initial architectures but may suffer from premature convergence or stagnation. Hybrid NEAT addresses this by:

1. **Adaptive Architecture Evolution**: Networks can dynamically adjust their topology during evolution, allowing them to discover optimal structures that weren't present in the initial population.

2. **Continuous Learning Capability**: Rather than relying solely on inherited "good genes," each individual maintains the ability to adapt and improve through structural modifications.

3. **Avoiding Stagnation**: The system prevents getting stuck in local optima by continuously introducing topological innovations and mutations that expand the search space.

4. **Maximizing Individual Potential**: Each network architecture is given the opportunity to reach its maximum potential through both genetic evolution and structural adaptation.

This approach ensures that the evolutionary process remains dynamic and exploratory, preventing the common problem where evolution converges too quickly to suboptimal solutions simply because they were "good enough" initially.

### Practical Benefits of Hybrid NEAT

This philosophy translates into several practical advantages:

1. **Higher Final Performance**: Networks that continue to adapt often achieve better final performance than those that start with good architectures but can't improve further.

2. **Robustness to Initial Conditions**: The system doesn't depend on having perfect initial architectures, making it more reliable across different problem domains.

3. **Discovery of Novel Solutions**: By maintaining adaptability, the system can discover unconventional but effective architectures that wouldn't emerge from traditional approaches.

4. **Scalability**: The adaptive nature allows the system to handle problems of varying complexity without requiring manual architecture design.

5. **Reduced Human Intervention**: The emphasis on adaptation reduces the need for domain expertise in neural architecture design.

> **Key Insight**: In neuroevolution, a network's ability to learn and adapt is more valuable than its initial performance. This is why Hybrid NEAT focuses on maintaining evolutionary pressure and diversity rather than just selecting the current best performers.

### Implementation in the System

The Hybrid NEAT approach is implemented through several key mechanisms:

- **Multi-generational Evolution**: The genetic algorithm runs for multiple generations, allowing architectures to continuously evolve and improve rather than stopping after a single evaluation.

- **Adaptive Mutation Rates**: The system uses intelligent mutation strategies that can modify network topology, layer sizes, activation functions, and other architectural parameters.

- **Fitness-based Selection with Diversity Preservation**: While selecting the best performers, the system maintains genetic diversity to prevent premature convergence to suboptimal solutions.

- **Incremental Complexity Growth**: Networks can start simple and gradually increase in complexity as needed, rather than beginning with overly complex architectures that may be hard to optimize.

The result is an evolutionary system that prioritizes **adaptability over initial perfection**, ensuring that each generation has the potential to surpass its predecessors through intelligent adaptation rather than just inheriting static "good" characteristics.

### Evolution Parameters

| Parameter | Description | MNIST | NEAT Impact |
|-----------|-------------|---------|-------------|
| `max_generations` | Maximum number of generations to evolve | 10 | 🔄 Allows continuous adaptation |
| `fitness_threshold` | Target fitness score (0.0-1.0) | 0.95 | 🎯 Prevents premature convergence |
| `num_poblation` | Population size | 10 | 🧬 Maintains genetic diversity |
| `num_channels` | Input channels (1=grayscale, 3=RGB) | 1 | 📊 Network input adaptation |
| `px_h`, `px_w` | Image dimensions | 28, 28 | 🖼️ Topology scaling capability |
| `num_classes` | Number of output classes | 10 | 🎯 Output layer adaptation |
| `batch_size` | Training batch size | 32 | ⚡ Training efficiency balance |

### Hybrid NEAT Specific Parameters

The system also supports additional parameters that leverage the Hybrid NEAT approach:

| Parameter | Description | Default | Purpose | Used on this proyect |
|-----------|-------------|---------|---------|---------|
| `mutation_rate` | Probability of genetic mutation | 0.1 | 🔄 Ensures continuous adaptation | Yes |

### Evolution Results

The genetic algorithm returns:
- **Best model UUID** and path
- **Generations completed**
- **Best fitness achieved**
- **Fitness history** across generations
- **Convergence reason** (threshold reached, max generations, stagnation)

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

## 🚀 Scripts de Lanzamiento

Se han creado varios scripts para facilitar el inicio del flujo de neuroevolución:

### Scripts Disponibles

1. **`start_flow.py`** - Script básico para lanzar el flujo con configuración por defecto
2. **`launch_neuroevolution_flow.py`** - Script completo con monitoreo de base de datos
3. **`check_database_status.py`** - Script para verificar el estado de la base de datos
4. **`docker_start_flow.py`** - Script optimizado para uso con Docker Compose

### Uso Rápido

```bash
# Lanzar flujo básico
python Broker/start_flow.py

# Lanzar con monitoreo completo
python Broker/launch_neuroevolution_flow.py

# Verificar estado de la base de datos
python Broker/check_database_status.py

# Para Docker Compose
python Broker/docker_start_flow.py
```

Consulta `Broker/LAUNCHER_README.md` para documentación detallada.