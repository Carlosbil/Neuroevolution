import logging
import colorlog
import uuid
import os
import json
import argparse
from confluent_kafka import Producer, Consumer, KafkaException, KafkaError
import signal
import sys
import time

# Configuración de variables de entorno
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
STORAGE_PATH = os.environ.get("BROKER_STORAGE_PATH", os.path.join(os.path.dirname(__file__), 'models'))

# Configuración de argumentos de línea de comandos
def parse_args():
    parser = argparse.ArgumentParser(description='Broker service for Neuroevolution')
    parser.add_argument('--storage-path', type=str, help='Path to store model files')
    return parser.parse_args()

# Obtener la ruta de almacenamiento configurada
def get_storage_path():
    args = parse_args()
    if args.storage_path:
        path = args.storage_path
    else:
        path = STORAGE_PATH
    
    # Asegurar que el directorio existe
    os.makedirs(path, exist_ok=True)
    return path

# Configuración global del logger, incluyendo el nombre del archivo y la línea
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Evitar mensajes duplicados: limpiar cualquier manejador existente y deshabilitar la propagación.
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

# Crear un manejador de log con colores para la salida en consola.
log_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
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
logger.addHandler(log_handler)


def check_initial_poblation(params):
    """ 
    Check the initial poblation parameters.
    """
    if 'num_channels' not in params or 'px_h' not in params or 'px_w' not in params or 'num_classes' not in params or 'batch_size' not in params or 'num_poblation' not in params:
        return False
    return True


def generate_uuid():
    """ 
    Generate a random UUID.
    """
    return str(uuid.uuid4())


def get_possible_models(models_uuid):
    """ 
    Get the possible models from the given UUID.
    """
    # Load possible models from the configured storage path
    path = os.path.join(get_storage_path(), f'{models_uuid}.json')
    with open(path, 'r') as file:
        possible_models = json.load(file)
    return possible_models


def produce_message(producer, topic, message, times=10):
    """Publish a message to Kafka."""
    for i in range(times):
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        logger.info(f"✅ Mensaje enviado: {message} - Tópico: {topic} - {i} ")

def create_producer():
    """Crea un productor de Kafka."""
    return Producer({
        'bootstrap.servers': KAFKA_BROKER,
        'linger.ms': 0,  # clave para envío inmediato
        'batch.size': 1, # opcional
        })


def create_consumer():
    """Crea y configura un consumidor de Kafka con mejor manejo de errores."""
    return Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'broker-consumer-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,  # Asegura que los offsets se confirmen automáticamente
        'session.timeout.ms': 60000,  # Evita desconexiones prematuras
        'heartbeat.interval.ms': 15000,  # Reduce el riesgo de expulsión por latencia
        'max.poll.interval.ms': 300000,  # Permite más tiempo para procesar mensajes
    })