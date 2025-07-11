#!/usr/bin/env python3
"""
Script para iniciar el flujo de neuroevolución desde el principio.
Este script:
1. Se conecta a la base de datos
2. Lanza el tópico inicial de Kafka para crear la población inicial
3. Monitorea la base de datos para verificar el progreso del proceso
4. Proporciona información sobre el estado del proceso
"""

import os
import sys
import time
import json
import signal
import argparse
from datetime import datetime
from confluent_kafka import Producer, Consumer, KafkaException, KafkaError
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import logging
import colorlog

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar logger con colores
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)
log_handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_handler)
logger.propagate = False

# Configuración de Kafka
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

# Configuración de PostgreSQL
PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_DB = os.environ.get("POSTGRES_DB", "neuroevolution")

# Control de shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    """Maneja las señales de interrupción para un shutdown limpio."""
    global shutdown_flag
    logger.info("🛑 Recibida señal de interrupción. Iniciando shutdown...")
    shutdown_flag = True

def get_db_connection():
    """
    Crea y retorna una conexión a la base de datos PostgreSQL.
    
    :return: Conexión a la base de datos
    :rtype: psycopg2.extensions.connection
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DB,
            connect_timeout=10,
            application_name='neuroevolution_launcher'
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        raise

def create_kafka_producer():
    """
    Crea y configura un productor de Kafka.
    
    :return: Productor de Kafka configurado
    :rtype: confluent_kafka.Producer
    """
    try:
        producer = Producer({
            'bootstrap.servers': KAFKA_BROKER,
            'linger.ms': 0,
            'batch.size': 1,
        })
        return producer
    except Exception as e:
        logger.error(f"❌ Error creando productor de Kafka: {e}")
        raise

def send_initial_population_message(producer, config, topic="create-initial-population"):
    """
    Envía el mensaje inicial para crear la población o ejecutar el algoritmo genético completo.
    
    :param producer: Productor de Kafka
    :type producer: confluent_kafka.Producer
    :param config: Configuración de la población inicial o algoritmo genético
    :type config: dict
    :param topic: Tópico de Kafka a usar
    :type topic: str
    """
    try:
        message = json.dumps(config)
        
        logger.info(f"🚀 Enviando mensaje inicial al tópico '{topic}'")
        logger.info(f"📋 Configuración: {config}")
        
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        
        logger.info(f"✅ Mensaje enviado exitosamente al tópico '{topic}'")
        return True
    except Exception as e:
        logger.error(f"❌ Error enviando mensaje inicial: {e}")
        return False

def check_database_status(conn):
    """
    Verifica el estado actual de la base de datos.
    
    :param conn: Conexión a la base de datos
    :type conn: psycopg2.extensions.connection
    :return: Diccionario con estadísticas de la base de datos
    :rtype: dict
    """
    try:
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # Contar poblaciones
            cursor.execute("SELECT COUNT(*) as count FROM populations")
            populations_count = cursor.fetchone()['count']
            
            # Contar modelos
            cursor.execute("SELECT COUNT(*) as count FROM models")
            models_count = cursor.fetchone()['count']
            
            # Obtener la población más reciente
            cursor.execute("""
                SELECT uuid, created_at 
                FROM populations 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            latest_population = cursor.fetchone()
            
            # Contar modelos por población más reciente
            models_in_latest = 0
            if latest_population:
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM models 
                    WHERE population_uuid = %s
                """, (latest_population['uuid'],))
                models_in_latest = cursor.fetchone()['count']
            
            # Obtener modelos con scores
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM models 
                WHERE score > 0
            """)
            models_with_scores = cursor.fetchone()['count']
            
            return {
                'populations_count': populations_count,
                'models_count': models_count,
                'latest_population': latest_population,
                'models_in_latest': models_in_latest,
                'models_with_scores': models_with_scores,
                'timestamp': datetime.now()
            }
    except Exception as e:
        logger.error(f"❌ Error verificando estado de la base de datos: {e}")
        return None

def monitor_database_progress(conn, initial_status, target_models=None):
    """
    Monitorea el progreso en la base de datos.
    
    :param conn: Conexión a la base de datos
    :type conn: psycopg2.extensions.connection
    :param initial_status: Estado inicial de la base de datos
    :type initial_status: dict
    :param target_models: Número objetivo de modelos esperados
    :type target_models: int
    :return: True si se detecta progreso, False si hay error
    :rtype: bool
    """
    try:
        current_status = check_database_status(conn)
        if not current_status:
            return False
        
        # Verificar cambios desde el estado inicial
        populations_added = current_status['populations_count'] - initial_status['populations_count']
        models_added = current_status['models_count'] - initial_status['models_count']
        
        if populations_added > 0 or models_added > 0:
            logger.info(f"📊 Progreso detectado:")
            logger.info(f"   - Poblaciones: {current_status['populations_count']} (+{populations_added})")
            logger.info(f"   - Modelos: {current_status['models_count']} (+{models_added})")
            logger.info(f"   - Modelos en última población: {current_status['models_in_latest']}")
            logger.info(f"   - Modelos con score: {current_status['models_with_scores']}")
            
            # Verificar si se ha completado el objetivo
            if target_models and current_status['models_in_latest'] >= target_models:
                logger.info(f"🎯 Objetivo alcanzado: {current_status['models_in_latest']}/{target_models} modelos")
                return True
        
        return False
    except Exception as e:
        logger.error(f"❌ Error monitoreando progreso: {e}")
        return False

def main():
    """Función principal del script."""
    # Configurar manejador de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Lanzador del flujo de neuroevolución')
    parser.add_argument('--num-channels', type=int, default=3, help='Número de canales (default: 3)')
    parser.add_argument('--px-h', type=int, default=32, help='Altura de píxeles (default: 32)')
    parser.add_argument('--px-w', type=int, default=32, help='Ancho de píxeles (default: 32)')
    parser.add_argument('--num-classes', type=int, default=10, help='Número de clases (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch (default: 32)')
    parser.add_argument('--num-poblation', type=int, default=10, help='Tamaño de la población (default: 10)')
    parser.add_argument('--max-generations', type=int, default=10, help='Número máximo de generaciones (default: 10)')
    parser.add_argument('--fitness-threshold', type=float, default=0.95, help='Umbral de fitness objetivo (default: 0.95)')
    parser.add_argument('--use-complete-ga', action='store_true', help='Usar algoritmo genético completo con múltiples generaciones')
    parser.add_argument('--monitor-timeout', type=int, default=300, help='Timeout de monitoreo en segundos (default: 300)')
    parser.add_argument('--poll-interval', type=int, default=10, help='Intervalo de polling en segundos (default: 10)')
    
    args = parser.parse_args()
    
    # Configuración de la población inicial o algoritmo genético completo
    config = {
        'num_channels': args.num_channels,
        'px_h': args.px_h,
        'px_w': args.px_w,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_poblation': args.num_poblation
    }
    
    # Agregar parámetros del algoritmo genético completo si se solicita
    if args.use_complete_ga:
        config.update({
            'max_generations': args.max_generations,
            'fitness_threshold': args.fitness_threshold
        })
    
    algorithm_type = "algoritmo genético completo" if args.use_complete_ga else "población inicial"
    logger.info(f"🚀 Iniciando lanzador del flujo de neuroevolución - {algorithm_type}")
    logger.info(f"📋 Configuración: {config}")
    
    try:
        # Conectar a la base de datos
        logger.info("🔗 Conectando a la base de datos...")
        conn = get_db_connection()
        logger.info("✅ Conexión a la base de datos establecida")
        
        # Verificar estado inicial
        logger.info("📊 Verificando estado inicial de la base de datos...")
        initial_status = check_database_status(conn)
        if not initial_status:
            logger.error("❌ No se pudo verificar el estado inicial de la base de datos")
            return 1
        
        logger.info(f"📈 Estado inicial - Poblaciones: {initial_status['populations_count']}, Modelos: {initial_status['models_count']}")
        
        # Crear productor de Kafka
        logger.info("🔗 Creando productor de Kafka...")
        producer = create_kafka_producer()
        logger.info("✅ Productor de Kafka creado")
        
        # Enviar mensaje inicial o algoritmo genético completo
        topic = "genetic-algorithm" if args.use_complete_ga else "create-initial-population"
        if not send_initial_population_message(producer, config, topic):
            logger.error("❌ No se pudo enviar el mensaje inicial")
            return 1
        
        # Monitorear progreso
        logger.info(f"👁️ Iniciando monitoreo de la base de datos (timeout: {args.monitor_timeout}s, intervalo: {args.poll_interval}s)")
        
        start_time = time.time()
        last_progress_time = start_time
        
        while not shutdown_flag:
            # Verificar timeout
            if time.time() - start_time > args.monitor_timeout:
                logger.warning(f"⏰ Timeout alcanzado ({args.monitor_timeout}s)")
                break
            
            # Verificar progreso
            if monitor_database_progress(conn, initial_status, config['num_poblation']):
                logger.info("🎉 Proceso completado exitosamente")
                break
            
            # Mostrar estado cada minuto
            if time.time() - last_progress_time > 60:
                current_status = check_database_status(conn)
                if current_status:
                    logger.info(f"⏱️ Estado actual - Poblaciones: {current_status['populations_count']}, Modelos: {current_status['models_count']}")
                last_progress_time = time.time()
            
            # Esperar antes del siguiente polling
            time.sleep(args.poll_interval)
        
        if shutdown_flag:
            logger.info("🛑 Proceso interrumpido por el usuario")
        
        logger.info("🏁 Finalizando lanzador del flujo de neuroevolución")
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso principal: {e}")
        return 1
    finally:
        # Limpiar recursos
        try:
            if 'conn' in locals():
                conn.close()
                logger.info("🔒 Conexión a la base de datos cerrada")
        except:
            pass
        
        try:
            if 'producer' in locals():
                producer.flush()
                logger.info("🔒 Productor de Kafka cerrado")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
