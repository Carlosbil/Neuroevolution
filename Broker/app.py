import json
import os
import logging
import requests
from confluent_kafka import Consumer, KafkaException, KafkaError
from topic_process.create_initial_population import process_create_initial_population
from topic_process.evaluate_population import process_evaluate_population
from topic_process.select_best_architectures import process_select_best_architectures
from responses_process.create_initial_population_response import process_create_initial_population_response
from responses_process.evaluate_population_response import process_evaluate_population_response
from database import init_db

from utils import (
    logger,
    check_initial_poblation,
    generate_uuid,
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
    "create-initial-population": process_create_initial_population,
    "evaluate-population": process_evaluate_population,
    "select-best-architectures": process_select_best_architectures,
    "genome-create-initial-population-response": process_create_initial_population_response,
    "evolutioner-create-cnn-model-response": process_evaluate_population_response,
}


def main():
    # Initialize database
    logger.info("🚀 Starting Neuroevolution Broker...")
    logger.info("🔧 Initializing database...")
    init_db()
    
    # Log available topics
    logger.info(f"📋 Available topics: {list(TOPIC_PROCESSORS.keys())}")
    
    consumer = create_kafka_consumer()
    logger.info("🎯 Started Kafka Consumer and subscribed to topics")
    logger.info("👂 Listening for messages...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"📍 Reached end of partition: {msg.topic()} [{msg.partition()}]")
                else:
                    logger.error(f"❌ Kafka error: {msg.error()}")
                continue

            topic = msg.topic()
            try:
                data = json.loads(msg.value().decode('utf-8'))
                logger.info(f"✅ Received message on topic '{topic}': {data}")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to decode JSON message on topic '{topic}': {e}")
                logger.error(f"📄 Raw message: {msg.value().decode('utf-8', errors='replace')}")
                continue

            processor = TOPIC_PROCESSORS.get(topic)
            if processor:
                try:
                    logger.info(f"🔄 Processing message with {processor.__name__}")
                    processor(topic, data)
                    logger.info(f"✅ Successfully processed message on topic '{topic}'")
                except Exception as e:
                    logger.error(f"❌ Error processing message on topic '{topic}': {e}")
                    logger.error(f"🔍 Error type: {type(e).__name__}")
                    logger.error(f"📄 Data that caused error: {data}")
            else:
                logger.error(f"❌ Unknown topic: {topic}")
                logger.error(f"📋 Available topics: {list(TOPIC_PROCESSORS.keys())}")

    except KeyboardInterrupt:
        logger.info("⚠️ Shutting down consumer via keyboard interrupt... ⚠️")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main loop: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
    finally:
        consumer.close()
        logger.info("🔒 Kafka consumer closed")


if __name__ == "__main__":
    main()
