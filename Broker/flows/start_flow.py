#!/usr/bin/env python3
"""
Script simple para lanzar el flujo de neuroevolución.
Uso básico: python start_flow.py
"""

import os
import sys
import json
import time
from confluent_kafka import Producer
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de Kafka
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

def create_producer():
    """Crea un productor de Kafka."""
    return Producer({
        'bootstrap.servers': KAFKA_BROKER,
        'linger.ms': 0,
        'batch.size': 1,
    })

def send_message(producer, topic, message):
    """Envía un mensaje a un tópico de Kafka."""
    try:
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        logger.info(f"✅ Mensaje enviado al tópico '{topic}': {message}")
        return True
    except Exception as e:
        logger.error(f"❌ Error enviando mensaje: {e}")
        return False

def main():
    """Función principal."""
    # Configuración por defecto para MNIST - usando algoritmo genético completo
    config = {
        'num_channels': 1,      # Grayscale
        'px_h': 28,            # Altura MNIST
        'px_w': 28,            # Ancho MNIST
        'num_classes': 10,     # Clases MNIST
        'batch_size': 32,      # Tamaño de batch
        'num_poblation': 4,    # Tamaño de población
        'fitness_threshold': 0.95,  # Umbral de fitness
        'max_generations': 3,  # Máximo de generaciones
        'mutation_rate': 0.1   # Tasa de mutación
    }
    
    logger.info("🚀 Iniciando algoritmo genético completo (SIN bucle infinito)")
    logger.info(f"📋 Configuración: {config}")
    logger.info("🛑 Criterios de parada automáticos:")
    logger.info(f"   - Máximo {config['max_generations']} generaciones")
    logger.info(f"   - Fitness objetivo: {config['fitness_threshold']}")
    
    try:
        # Crear productor
        producer = create_producer()
        
        # Enviar mensaje al algoritmo genético completo
        topic = "genetic-algorithm"  # Cambiado de "create-initial-population"
        message = json.dumps(config)
        
        if send_message(producer, topic, message):
            logger.info("🎉 Algoritmo genético iniciado correctamente")
            logger.info("💡 El algoritmo se detendrá automáticamente cuando:")
            logger.info("   • Se alcance el fitness objetivo")
            logger.info("   • Se complete el número máximo de generaciones")
            logger.info("   • Se detecte estancamiento en el fitness")
            logger.info("📊 Puedes monitorear el progreso en los logs del Broker")
        else:
            logger.error("❌ Error iniciando el algoritmo genético")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
