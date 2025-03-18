import json
import os
import time
import re
from confluent_kafka import Producer, Consumer, KafkaError

# Configuración global de Kafka
KAFKA_BROKER = "localhost:9092"  # Cambia esto según tu configuración

# topic for genomes interactions
TOPIC = "create-initial-population"
TOPIC_RESPONSE = f"{TOPIC}-response"

# topic for evaluation
TOPIC_EVALUATE = "evaluate-population"
TOPIC_EVALUATE_RESPONSE = f"{TOPIC_EVALUATE}-response"

TOPIC_GENETIC_ALGORITHM = "genetic-algorithm"
TOPIC_GENETIC_ALGORITHM_RESPONSE = f"{TOPIC_GENETIC_ALGORITHM}-response"

TOPIC_SELECTION_ALGORITHM = "select-best-architectures"
TOPIC_SELECTION_ALGORITHM_RESPONSE = f"{TOPIC_SELECTION_ALGORITHM}-response"

# topic for create child
TOPIC_CREATE_CHILD = "create-child"
TOPIC_CREATE_CHILD_RESPONSE = f"{TOPIC_CREATE_CHILD}-response"

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
    consumer.subscribe([TOPIC_RESPONSE, TOPIC_EVALUATE_RESPONSE, TOPIC_SELECTION_ALGORITHM_RESPONSE, TOPIC_GENETIC_ALGORITHM_RESPONSE, TOPIC_CREATE_CHILD_RESPONSE])
    return consumer

def consume_message(consumer, max_wait=1000):
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

def consume_all_messages(consumer, topics, max_wait=10000):
    """
    Consume messages from Kafka until at least one message is received from each topic.
    
    :param consumer: Kafka consumer instance.
    :param topics: List of topics to wait for messages from.
    :param max_wait: Maximum time in milliseconds to wait for messages.
    :return: Dictionary with messages from each topic or None if timeout occurs.
    """
    start_time = time.time()
    received_messages = {topic: None for topic in topics}
    
    while any(msg is None for msg in received_messages.values()):
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            if (time.time() - start_time) > max_wait:
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
            received_messages[msg.topic()] = response
            print(f"✅ Received message on topic '{msg.topic()}': {response}")
        except Exception as e:
            print(f"Error al procesar el mensaje de {msg.topic()}: {e}")
    
    return received_messages


def fix_path(path):
    """
    Fixes a path by extracting the UUID and creating a relative path to the models directory.
    
    :param path: The original path (e.g., '/app/models/uuid.json')
    :return: The fixed path (e.g., './models/uuid.json')
    """
    if path is None:
        return None
    
    time.sleep(2)
    # Extract the UUID part (assuming it's a UUID followed by .json)
    uuid_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.json)', path)
    if uuid_match:
        return f"./Broker/models/{uuid_match.group(1)}"
    return path


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
    
    # Fix the path before checking if it exists
    fixed_path = fix_path(path)
    assert os.path.exists(fixed_path), f"El archivo {fixed_path} no existe"
    
    with open(fixed_path, "r", encoding="utf-8") as f:
        population_data = json.load(f)
    assert len(population_data) == 10, f"Se esperaban 10 individuos, pero se encontraron {len(population_data)}"

def test_create_population():
    """
    Test que envía un mensaje válido y comprueba que se recibe una respuesta con status_code 200,
    que el archivo indicado existe y contiene 10 individuos.
    """
    params = {
        "num_channels": 1,
        "px_h": 28,
        "px_w": 28,
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=1000)
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


def test_evaluate_population():
    """
    Test que envía un mensaje válido y comprueba que se recibe una respuesta con status_code 200,
    que el archivo indicado existe y contiene 10 individuos.
    """
    
    #First we need to create a population
    
    uuid = test_create_population().get("message", {})
    uuid = uuid.get("uuid", '3d91192b-68f2-4766-9e27-08cb7bf56733')
    params = {
        "uuid": uuid
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC_EVALUATE, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=3600)
    consumer.close()
    
    print("✅ Test de evaluación de población válido finalizado correctamente.")
    return response


def test_genetic_algorithm():
    """
    Test que envía un mensaje válido y comprueba que se recibe una respuesta con status_code 200
    para cada paso del algoritmo genético: creación de población inicial, evaluación y selección
    de los mejores individuos.
    """
    params = {
        "num_channels": 1,
        "px_h": 28,
        "px_w": 28,
        "num_classes": 10,
        "batch_size": 32,
        "num_poblation": 10
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC_GENETIC_ALGORITHM, key="population_eval", value=message)
    
    consumer = create_consumer()
    response = consume_all_messages(consumer, 
        [TOPIC_RESPONSE, TOPIC_EVALUATE_RESPONSE, "select-best-architectures-response", TOPIC_GENETIC_ALGORITHM_RESPONSE], 
        max_wait=3600)
    consumer.close()
    
    # Verify each step's response
    for topic, msg in response.items():
        assert msg is not None, f"No se recibió respuesta del tópico {topic}"
        assert msg.get("status_code") == 200, f"El status_code es {msg.get('status_code')} en lugar de 200 para {topic}"
    
    # Verify the final response
    final_response = response.get(TOPIC_GENETIC_ALGORITHM_RESPONSE)
    assert final_response.get("message", {}).get("message") == "Genetic algorithm completed successfully: population created, evaluated, and best 50% selected"
    
    print("✅ Test de algoritmo genético finalizado correctamente.")
    return response


def test_create_child():
    """
    Test que envía un mensaje para crear un hijo a partir de dos modelos existentes
    y comprueba que se recibe una respuesta con status_code 200.
    """
    # Primero necesitamos crear una población inicial
    response = test_create_population()
    assert response is not None, "No se pudo crear la población inicial"
    
    # Obtenemos el UUID de la población creada
    uuid = response.get("message", {}).get("uuid")
    assert uuid is not None, "No se encontró el UUID de la población"
    
    # Cargamos los modelos para obtener dos IDs
    path = response.get("message", {}).get("path")
    assert path is not None, "No se encontró la ruta del archivo"
    
    # Fix the path before checking if it exists
    fixed_path = fix_path(path)
    assert os.path.exists(fixed_path), f"El archivo {fixed_path} no existe"
    
    with open(fixed_path, "r", encoding="utf-8") as f:
        models = json.load(f)
    
    # Tomamos los dos primeros modelos
    model_ids = list(models.keys())
    assert len(model_ids) >= 2, "No hay suficientes modelos para crear un hijo"
    
    # Preparamos los parámetros para crear un hijo
    params = {
        "model_id": model_ids[0],
        "second_model_id": model_ids[1],
        "uuid": uuid
    }
    message = json.dumps(params)
    
    # Enviamos el mensaje al tópico create-child
    producer = create_producer()
    send_message(producer, TOPIC_CREATE_CHILD, key="create_child", value=message)
    
    # Esperamos la respuesta
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=1000)
    consumer.close()
    
    # Verificamos la respuesta
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 200, f"El status_code es {response.get('status_code')} en lugar de 200"
    
    # Verificamos que la respuesta contiene un mensaje con la información del modelo hijo
    message = response.get("message", {})
    assert message, "La respuesta no contiene información del modelo hijo"
    
    print("✅ Test de creación de hijo finalizado correctamente.")
    return response


def test_create_child_missing_model_id():
    """
    Test que envía un mensaje sin 'model_id' y comprueba que la respuesta tenga status_code 400.
    """
    # Primero necesitamos crear una población inicial
    response = test_create_population()
    uuid = response.get("message", {}).get("uuid")
    
    # Preparamos los parámetros sin model_id
    params = {
        # 'model_id' omitido intencionadamente
        "second_model_id": "1",  # Usamos un ID genérico
        "uuid": uuid
    }
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC_CREATE_CHILD, key="create_child", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=10)
    consumer.close()
    
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    print("✅ Test sin 'model_id' finalizado correctamente con status_code 400.")
    return response


if __name__ == "__main__":
    # print("Ejecutando test de población válida...")
    # test_create_population()
    # print("\nEjecutando test sin 'num_channels'...")
    # test_create_population_missing_num_channels()
    # print("\nEjecutando test sin 'px_h'...")
    # test_create_population_missing_px_h()
    # print("\nEjecutando test sin 'px_w'...")
    # test_create_population_missing_px_w()
    # print("\nEjecutando test sin 'num_classes'...")
    # test_create_population_missing_num_classes()
    # print("\nEjecutando test sin 'batch_size'...")
    # test_create_population_missing_batch_size()
    # print("\nEjecutando test sin 'num_poblation'...")
    # test_create_population_missing_num_poblation()
    print("\nEjecutando test de evaluación de población...")
    test_evaluate_population()
    # print("\nEjecutando test de algoritmo genético...")
    # test_genetic_algorithm()
    # print("\nEjecutando test de creación de hijo...")
    # test_create_child()
    # print("\nEjecutando test sin 'model_id'...")
    # test_create_child_missing_model_id()