#!/usr/bin/env python3
"""
Script para iniciar el algoritmo genético completo con criterios de parada.
Este script evita el bucle infinito usando el algoritmo genético controlado.
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
    """Función principal para iniciar el algoritmo genético controlado."""
    # Configuración con criterios de parada
    config = {
        'num_channels': 1,           # Grayscale para MNIST
        'px_h': 28,                  # Altura MNIST
        'px_w': 28,                  # Ancho MNIST
        'num_classes': 10,           # Clases MNIST
        'batch_size': 32,            # Tamaño de batch
        'num_poblation': 6,          # Tamaño de población
        'max_generations': 5,        # CRITERIO DE PARADA: Máximo 5 generaciones
        'fitness_threshold': 99,   # CRITERIO DE PARADA: Detener si se alcanza 99% de precisión
        'mutation_rate': 0.1         # Tasa de mutación
    }
    
    logger.info("🚀 Iniciando algoritmo genético controlado")
    logger.info(f"📋 Configuración: {config}")
    logger.info("🛑 Criterios de parada configurados:")
    logger.info(f"   - Máximo {config['max_generations']} generaciones")
    logger.info(f"   - Fitness objetivo: {config['fitness_threshold']}")
    
    try:
        # Crear productor
        producer = create_producer()
        
        # Usar el tópico del algoritmo genético completo
        topic = "genetic-algorithm"
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
