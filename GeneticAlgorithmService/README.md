# Genetic Algorithm Service

## Overview

The Genetic Algorithm Service is a dedicated microservice that handles the complete genetic algorithm workflow for neural network evolution. This service was separated from the main Broker to prevent blocking operations and improve system scalability.

## Features

- **Asynchronous Processing**: Runs genetic algorithm operations without blocking the main broker
- **Multi-generation Evolution**: Manages complete evolutionary cycles with convergence criteria
- **Broker Communication**: Communicates with the main broker through Kafka topics
- **Resource Management**: Prevents overwhelming the system with controlled delays and timeouts

## Architecture

The service operates independently and communicates with other system components through Kafka:

```
Client -> Broker -> GeneticAlgorithmService -> Broker -> Genome/Evolutioners
```

## Topics Handled

- **Input**: `genetic-algorithm`
- **Output**: `genetic-algorithm-response`

## Communication Topics with Broker

The service communicates with the broker through these topics:

- `create-initial-population` / `create-initial-population-response`
- `evaluate-population` / `evaluate-population-response`
- `select-best-architectures` / `select-best-architectures-response`
- `create-child` / `create-child-response`
- `check-population` / `check-population-response`
- `check-evaluation` / `check-evaluation-response`
- `get-best-fitness` / `get-best-fitness-response`

## Process Flow

1. **Initialization**: Creates initial population through broker
2. **Evolution Loop**: For each generation:
   - Evaluates population fitness
   - Checks convergence criteria
   - Selects best individuals
   - Creates offspring through crossover/mutation
   - Replaces population with children
3. **Completion**: Returns final results with statistics

## Configuration

Environment variables:
- `KAFKA_BROKER`: Kafka broker address (default: `localhost:9092`)

## Running the Service

### Development
```bash
cd GeneticAlgorithmService
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker-compose up genetic-algorithm
```

## Benefits of Separation

1. **Non-blocking**: Broker remains responsive while genetic algorithm runs
2. **Scalability**: Can be scaled independently based on demand
3. **Fault Tolerance**: Isolated failures don't affect other broker operations
4. **Resource Management**: Dedicated resources for computationally intensive operations
5. **Maintainability**: Cleaner separation of concerns

## Error Handling

The service includes comprehensive error handling:
- Timeout management for broker communications
- Retry logic for failed operations
- Graceful degradation on failures
- Detailed logging for debugging

## Monitoring

The service provides detailed logging including:
- Generation progress
- Fitness evolution
- Convergence metrics
- Error tracking
- Performance statistics
