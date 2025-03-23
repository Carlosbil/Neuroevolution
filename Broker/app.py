import json
import os
import logging
import requests
from confluent_kafka import Consumer, KafkaException, KafkaError
from topic_process.create_initial_population import process_create_initial_population
from topic_process.evaluate_population import process_evaluate_population
from topic_process.create_child import process_create_child
from topic_process.genetic_algorithm import process_genetic_algorithm
from topic_process.select_best_architectures import process_select_best_architectures
from responses_process.create_initial_population_response import process_create_initial_population_response
from responses_process.evaluate_population_response import process_evaluate_population_response
from responses_process.create_child_response import process_create_child_response
from database import init_db, import_json_models

from utils import (
    logger,
    check_initial_poblation,
    generate_uuid,
    get_possible_models,
    create_producer,
    produce_message,
    create_consumer
)
from responses import (
    ok_message,
    bad_model_message,
    bad_optimizer_message,
    runtime_error_message,
    response_message,
    bad_request_message,
)

# Kafka configuration
# Function to create a Kafka consumer
def create_kafka_consumer():
    consumer = create_consumer()
    consumer.subscribe(list(TOPIC_PROCESSORS.keys()))  # Subscribe to all topics
    return consumer


# Dictionary-based topic-to-function mapping
TOPIC_PROCESSORS = {
    "create-child": process_create_child,
    "create-initial-population": process_create_initial_population,
    "evaluate-population": process_evaluate_population,
    "start-hybrid-neat": process_genetic_algorithm,
    "select-best-architectures": process_select_best_architectures,
    "genome-create-initial-population-response": process_create_initial_population_response,
    "evolutioner-create-cnn-model-response": process_evaluate_population_response,
    "genome-create-child-response": process_create_child_response,
}


def main():
    # Initialize database and import existing models
    logger.info("Initializing database...")
    init_db()
    logger.info("Importing existing JSON models...")
    import_json_models()
    
    consumer = create_kafka_consumer()
    logger.info("Started Kafka Consumer")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Reached end of partition: {msg.topic()} [{msg.partition()}]")
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                continue

            topic = msg.topic()
            data = json.loads(msg.value().decode('utf-8'))
            logger.info(f" ✅ Received message on topic '{topic}': {data}")

            processor = TOPIC_PROCESSORS.get(topic)
            if processor:
                processor(topic, data)
            else:
                logger.error(f"Unknown topic: {topic}")

    except KeyboardInterrupt:
        logger.error("⚠️ Shutting down consumer via keyboard... ⚠️")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
