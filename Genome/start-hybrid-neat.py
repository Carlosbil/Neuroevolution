from utils import logger, create_producer, produce_message, create_consumer
from responses import ok_message, bad_model_message, runtime_error_message, bad_request_message
import requests
import json
import os
from confluent_kafka import KafkaError


def process_start_hybrid_neat(topic, data = {}):
    """Process messages from the 'start-hybrid-neat' topic and launch the complete genetic algorithm."""
    try:
        logger.info("🚀 Starting Hybrid NEAT - Complete Genetic Algorithm")
        
        # Default parameters for NEAT-style evolution
        default_data = {
            "num_channels": 1,
            "px_h": 28,
            "px_w": 28,
            "num_classes": 10,
            "batch_size": 32,
            "num_poblation": 5,
            "max_generations": 4,  # Allow for multiple generations
            "fitness_threshold": 0.95  # Target 95% accuracy
        }
        
        # Merge with provided data
        if isinstance(data, dict):
            default_data.update(data)
        
        logger.info(f"🎯 NEAT Parameters: generations={default_data['max_generations']}, "
                   f"population={default_data['num_poblation']}, "
                   f"fitness_threshold={default_data['fitness_threshold']}")
        
        # Send to genetic algorithm topic for complete evolution
        producer = create_producer()
        topic_to_send = "genetic-algorithm"
        produce_message(producer, topic_to_send, json.dumps(default_data), times=1)
        
        logger.info(f"✅ Successfully launched complete genetic algorithm from Hybrid NEAT")
        
        return ok_message(topic, {
            "message": "Hybrid NEAT genetic algorithm launched successfully",
            "parameters": default_data,
            "status": "processing"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in start_hybrid_neat: {e}")
        return runtime_error_message(topic)
    
process_start_hybrid_neat("start-hybrid-neat")