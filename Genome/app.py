import json
import os
import logging
import requests
from confluent_kafka import Consumer, KafkaException, KafkaError
from topic_process.create_child import handle_create_child
from topic_process.create_initial_population import handle_create_initial_population

from utils import (
    logger,
)


# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'consumer-group',
    'auto.offset.reset': 'earliest'
}

# Function to create a Kafka consumer
def create_kafka_consumer():
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe(list(TOPIC_PROCESSORS.keys()))  # Subscribe to all topics
    return consumer


# Dictionary-based topic-to-function mapping
TOPIC_PROCESSORS = {
    "genome-create-child": handle_create_child,
    "genome-create-initial-population": handle_create_initial_population,
}


def main():
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
