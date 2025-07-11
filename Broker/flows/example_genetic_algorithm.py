#!/usr/bin/env python3
"""
Ejemplo de ejecución del algoritmo genético completo para evolución de redes neuronales.
Este script muestra cómo configurar y ejecutar el algoritmo genético con múltiples generaciones.
"""

import json
import time
from confluent_kafka import Producer, Consumer, KafkaError
from utils import create_producer, create_consumer, logger

def send_genetic_algorithm_request(parameters):
    """
    Envía una solicitud al algoritmo genético con los parámetros especificados.
    
    Args:
        parameters (dict): Parámetros del algoritmo genético
    """
    try:
        producer = create_producer()
        topic = "genetic-algorithm"
        message = json.dumps(parameters)
        
        logger.info(f"🚀 Enviando solicitud al algoritmo genético...")
        logger.info(f"📊 Parámetros: {parameters}")
        
        producer.produce(topic, message.encode('utf-8'))
        producer.flush()
        
        logger.info(f"✅ Solicitud enviada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error enviando solicitud: {e}")
        return False

def monitor_genetic_algorithm_progress():
    """
    Monitorea el progreso del algoritmo genético y muestra los resultados.
    """
    try:
        consumer = create_consumer()
        consumer.subscribe(["genetic-algorithm-response"])
        
        logger.info("👂 Monitoreando progreso del algoritmo genético...")
        
        start_time = time.time()
        timeout = 7200  # 2 horas máximo
        
        while time.time() - start_time < timeout:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"❌ Error de Kafka: {msg.error()}")
                    break
            
            try:
                data = json.loads(msg.value().decode('utf-8'))
                logger.info(f"📨 Respuesta recibida: {data}")
                
                if data.get("status_code") == 200:
                    result = data.get("message", {})
                    
                    logger.info("🎉 Algoritmo genético completado exitosamente!")
                    logger.info(f"🔄 Generaciones completadas: {result.get('generations_completed', 'N/A')}")
                    logger.info(f"🏆 Mejor fitness: {result.get('best_fitness', 'N/A')}")
                    logger.info(f"📈 Historial de fitness: {result.get('fitness_history', [])}")
                    logger.info(f"✅ Razón de convergencia: {result.get('convergence_reason', 'N/A')}")
                    logger.info(f"📁 UUID del mejor modelo: {result.get('uuid', 'N/A')}")
                    
                    return result
                else:
                    logger.error(f"❌ Error en algoritmo genético: {data}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error decodificando respuesta: {e}")
                continue
        
        logger.warning("⏰ Timeout alcanzado esperando respuesta del algoritmo genético")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error monitoreando algoritmo genético: {e}")
        return None
    finally:
        consumer.close()

def main():
    """Función principal que ejecuta el ejemplo completo."""
    logger.info("🧬 Iniciando ejemplo del algoritmo genético completo")
    
    # Configuración del algoritmo genético
    genetic_params = {
        # Parámetros del dataset
        "num_channels": 1,      # 1 para MNIST (escala de grises)
        "px_h": 28,            # Altura de imagen
        "px_w": 28,            # Ancho de imagen
        "num_classes": 10,     # 10 clases para MNIST
        "batch_size": 32,      # Tamaño de lote
        
        # Parámetros evolutivos
        "num_poblation": 8,    # Población pequeña para ejemplo
        "max_generations": 5,  # Pocas generaciones para ejemplo
        "fitness_threshold": 0.9,  # Meta de 90% de accuracy
    }
    
    # Enviar solicitud
    if not send_genetic_algorithm_request(genetic_params):
        logger.error("❌ No se pudo enviar la solicitud")
        return 1
    
    # Monitorear progreso
    result = monitor_genetic_algorithm_progress()
    
    if result:
        logger.info("🎯 Ejemplo completado exitosamente!")
        return 0
    else:
        logger.error("❌ El ejemplo no se completó correctamente")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
