import json
import os
import logging
import requests
from confluent_kafka import Consumer, KafkaException, KafkaError
from topic_process.genetic_algorithm import handle_genetic_algorithm, handle_continue_algorithm

from utils import (
    logger,
    create_consumer
)

# Dictionary-based topic-to-function mapping
TOPIC_PROCESSORS = {
    "genetic-algorithm": handle_genetic_algorithm,
    "continue-algorithm": handle_continue_algorithm,
}

# Function to create a Kafka consumer
def create_kafka_consumer():
    consumer = create_consumer()
    consumer.subscribe(list(TOPIC_PROCESSORS.keys()))  # Subscribe to all topics
    return consumer

def main():
    # Log available topics
    logger.info(f"🧬 Starting Genetic Algorithm Service...")
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
        logger.error("⚠️ Shutting down consumer via keyboard... ⚠️")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
