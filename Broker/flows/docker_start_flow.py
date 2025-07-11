#!/usr/bin/env python3
"""
Script para usar con Docker Compose.
Este script espera a que los servicios estén listos antes de lanzar el flujo.
"""

import os
import sys
import time
import json
import socket
import psycopg2
from confluent_kafka import Producer, KafkaException
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
PG_HOST = os.environ.get("POSTGRES_HOST", "postgres")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_DB = os.environ.get("POSTGRES_DB", "neuroevolution")

def wait_for_service(host, port, timeout=60):
    """Espera a que un servicio esté disponible."""
    logger.info(f"⏳ Esperando a que {host}:{port} esté disponible...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ {host}:{port} está disponible")
                return True
                
        except socket.error:
            pass
        
        time.sleep(1)
    
    logger.error(f"❌ Timeout esperando a {host}:{port}")
    return False

def wait_for_kafka():
    """Espera a que Kafka esté disponible."""
    # Extraer host y puerto de KAFKA_BROKER
    try:
        if ":" in KAFKA_BROKER:
            host, port = KAFKA_BROKER.split(":")
            port = int(port)
        else:
            host = KAFKA_BROKER
            port = 9092
        
        if not wait_for_service(host, port):
            return False
        
        # Verificar que podemos crear un productor
        logger.info("🔍 Verificando conectividad con Kafka...")
        try:
            producer = Producer({'bootstrap.servers': KAFKA_BROKER})
            producer.list_topics(timeout=5)
            logger.info("✅ Kafka está listo")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando a Kafka: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verificando Kafka: {e}")
        return False

def wait_for_postgres():
    """Espera a que PostgreSQL esté disponible."""
    if not wait_for_service(PG_HOST, PG_PORT):
        return False
    
    # Verificar que podemos conectar a la base de datos
    logger.info("🔍 Verificando conectividad con PostgreSQL...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DB,
                connect_timeout=5
            )
            conn.close()
            logger.info("✅ PostgreSQL está listo")
            return True
        except Exception as e:
            attempt += 1
            logger.info(f"🔄 Intento {attempt}/{max_attempts} - PostgreSQL no está listo: {e}")
            time.sleep(2)
    
    logger.error("❌ PostgreSQL no está disponible después de múltiples intentos")
    return False

def send_initial_message():
    """Envía el mensaje inicial para crear la población."""
    try:
        config = {
            'num_channels': 1,      # Grayscale
            'px_h': 28,            # Altura MNIST
            'px_w': 28,            # Ancho MNIST
            'num_classes': 10,     # Clases MNIST
            'batch_size': 32,      # Tamaño de batch
            'num_poblation': 10    # Tamaño de población
        }
        
        producer = Producer({'bootstrap.servers': KAFKA_BROKER})
        topic = "create-initial-population"
        message = json.dumps(config)
        
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        
        logger.info(f"✅ Mensaje enviado al tópico '{topic}'")
        logger.info(f"📋 Configuración: {config}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error enviando mensaje inicial: {e}")
        return False

def main():
    """Función principal."""
    logger.info("🚀 Iniciando lanzador para Docker Compose")
    
    # Esperar a que los servicios estén listos
    logger.info("⏳ Esperando a que los servicios estén listos...")
    
    if not wait_for_postgres():
        logger.error("❌ PostgreSQL no está disponible")
        return 1
    
    if not wait_for_kafka():
        logger.error("❌ Kafka no está disponible")
        return 1
    
    # Esperar un poco más para asegurar que los servicios estén completamente listos
    logger.info("⏱️ Esperando 10 segundos adicionales para estabilización...")
    time.sleep(10)
    
    # Enviar mensaje inicial
    if send_initial_message():
        logger.info("🎉 Flujo iniciado correctamente")
        logger.info("💡 Monitorea los logs de los servicios para ver el progreso")
        return 0
    else:
        logger.error("❌ Error iniciando el flujo")
        return 1

if __name__ == "__main__":
    sys.exit(main())
