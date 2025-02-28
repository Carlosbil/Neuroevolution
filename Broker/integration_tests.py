import json
import os
import time
from confluent_kafka import Producer, Consumer, KafkaError

# Configuración global de Kafka
KAFKA_BROKER = "localhost:9092"  # Cambia esto según tu configuración
TOPIC = "create-initial-population"
TOPIC_RESPONSE = "create-initial-population-response"

def create_producer():
    producer_config = {"bootstrap.servers": KAFKA_BROKER}
    return Producer(producer_config)

def send_message(producer, topic, key, value):
    def delivery_report(err, msg):
        if err:
            print(f"❌ Error al enviar mensaje: {err}")
        else:
            print(f"✅ Mensaje enviado a {msg.topic()} [{msg.partition()}]")
    producer.produce(topic, key=key, value=value, callback=delivery_report)
    producer.flush()

def create_consumer(group_id='consumer-group'):
    consumer_config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([TOPIC_RESPONSE])
    return consumer

def consume_message(consumer, max_wait=10):
    start_time = time.time()
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            if time.time() - start_time > max_wait:
                print("⏰ Tiempo máximo de espera alcanzado.")
                return None
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"Reached end of partition: {msg.topic()} [{msg.partition()}]")
            else:
                print(f"Error: {msg.error()}")
            continue
        try:
            response = json.loads(msg.value().decode('utf-8'))
            print(f"✅ Received message on topic '{msg.topic()}': {response}")
            return response
        except Exception as e:
            print(f"Error al procesar el mensaje: {e}")

def check_response(response):
    """
    Verifica que la respuesta tenga status_code 200, que el archivo exista
    y que contenga 10 individuos.
    """
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 200, f"El status_code es {response.get('status_code')} en lugar de 200"
    
    message = response.get("message", {})
    path = message.get("path")
    assert path is not None, "No se encontró la ruta del archivo en la respuesta"
    assert os.path.exists(path), f"El archivo {path} no existe"
    
    with open(path, "r", encoding="utf-8") as f:
        population_data = json.load(f)
    assert len(population_data) == 10, f"Se esperaban 10 individuos, pero se encontraron {len(population_data)}"

def test_create_population():
    """
    Test que envía un mensaje válido y comprueba que se recibe una respuesta con status_code 200,
    que el archivo indicado existe y contiene 10 individuos.
    """
    params = {
        "num_channels": 3,
        "px_h": 32,
        "px_w": 32,
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    check_response(response)
    print("✅ Test de creación de población válido finalizado correctamente.")
    return response

def test_create_population_missing_num_channels():
    """
    Test que envía un mensaje sin 'num_channels' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        # 'num_channels' omitido intencionadamente
        "px_h": 32,
        "px_w": 32,
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'num_channels' finalizado correctamente con status_code 400.")
    return response

def test_create_population_missing_px_h():
    """
    Test que envía un mensaje sin 'px_h' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        'num_channels': 3,
        # 'px_h' omitido intencionadamente
        "px_w": 32,
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'num_channels' finalizado correctamente con status_code 400.")
    return response

def test_create_population_missing_px_w():
    """
    Test que envía un mensaje sin 'px_w' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        'num_channels': 3,
        'px_h': 32,
        # 'px_w' omitido intencionadamente
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'px_w' finalizado correctamente con status_code 400.")
    return response

def test_create_population_missing_num_classes():
    """
    Test que envía un mensaje sin 'num_classes' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        'num_channels': 3,
        'px_h': 32,
        'px_w': 32,
        # 'num_classes' omitido intencionadamente
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'num_classes' finalizado correctamente con status_code 400.")
    return response

def test_create_population_missing_batch_size():
    """
    Test que envía un mensaje sin 'batch_size' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        'num_channels': 3,
        'px_h': 32,
        'px_w': 32,
        'num_classes': 10,
        # 'batch_size' omitido intencionadamente
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'batch_size' finalizado correctamente con status_code 400.")
    return response

def test_create_population_missing_num_poblation():
    """
    Test que envía un mensaje sin 'num_poblation' y comprueba que la respuesta tenga status_code 400.
    """
    params = {
        'num_channels': 3,
        'px_h': 32,
        'px_w': 32,
        'num_classes': 10,
        'batch_size': 32,
        # 'num_poblation' omitido intencionadamente
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'num_poblation' finalizado correctamente con status_code 400.")
    return response
    
if __name__ == "__main__":
    print("Ejecutando test de población válida...")
    test_create_population()
    print("\nEjecutando test sin 'num_channels'...")
    test_create_population_missing_num_channels()
    print("\nEjecutando test sin 'px_h'...")
    test_create_population_missing_px_h()
    print("\nEjecutando test sin 'px_w'...")
    test_create_population_missing_px_w()
    print("\nEjecutando test sin 'num_classes'...")
    test_create_population_missing_num_classes()
    print("\nEjecutando test sin 'batch_size'...")
    test_create_population_missing_batch_size()
    print("\nEjecutando test sin 'num_poblation'...")
    test_create_population_missing_num_poblation()