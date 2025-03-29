import json
import time
from confluent_kafka import Producer, Consumer, KafkaError
from topic_process.create_cnn_model import handle_create_cnn_model

# Configuración global de Kafka
import os
import dotenv
dotenv.load_dotenv()

# Use environment variables with fallbacks
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPIC = "evolutioner-create-cnn-model"
TOPIC_RESPONSE = f"{TOPIC}-response"

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

def create_consumer(group_id='integration_consumer'):
    consumer_config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([TOPIC_RESPONSE])
    return consumer

def consume_message(consumer, max_wait=30):
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
                continue
            else:
                print(f"Error: {msg.error()}")
                continue
        try:
            response = json.loads(msg.value().decode('utf-8'))
            print(f"✅ Mensaje recibido en '{msg.topic()}': {response}")
            return response
        except Exception as e:
            print(f"Error al procesar el mensaje: {e}")

def check_success_response(response):
    """
    Verifica que la respuesta tenga status_code 200 y que el mensaje interno contenga:
      - 'message': "CNN model created, trained and evaluated successfully"
      - 'score': valor numérico
    """
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 200, f"Se esperaba status_code 200, pero se recibió {response.get('status_code')}"
    
    inner_message = response.get("message")
    assert isinstance(inner_message, dict), "El contenido de 'message' debe ser un diccionario"
    expected_msg = "CNN model created, trained and evaluated successfully"
    assert inner_message.get("message") == expected_msg, (
        f"Se esperaba el mensaje '{expected_msg}', pero se recibió '{inner_message.get('message')}'"
    )
    
    score = inner_message.get("score")
    assert isinstance(score, (int, float)), "El score debe ser numérico"

def check_error_response(response, expected_error="Invalid model"):
    """
    Verifica que la respuesta tenga status_code 400 y el error esperado.
    """
    assert response is not None, "No se recibió respuesta del tópico"
    assert response.get("status_code") == 400, f"Se esperaba status_code 400, pero se recibió {response.get('status_code')}"
    assert response.get("error") == expected_error, (
        f"Se esperaba error '{expected_error}', pero se recibió '{response.get('error')}'"
    )

# Parámetros válidos según la configuración correcta del modelo.
def get_valid_params():
    return {
        "Number of Convolutional Layers": 2,
        "Number of Fully Connected Layers": 2,
        "Number of Nodes in Each Layer": [32],
        "Activation Functions": ["relu"],
        "Dropout Rate": 0.2,
        "Learning Rate": 0.0001,
        "Batch Size": 32,
        "Optimizer": "adamw",
        "num_channels": 1,
        "px_h": 28,
        "px_w": 28,
        "num_classes": 10,
        "filters": [3, 3],
        "kernel_sizes": [3, 3]
    }

def test_create_cnn_model_valid():
    """
    Test de integración para la creación exitosa del modelo CNN.
    Envía el mensaje con todos los parámetros válidos y se espera una respuesta exitosa.
    """
    params = get_valid_params()
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=600)
    consumer.close()
    
    check_success_response(response)
    print("✅ Test de creación de CNN válido finalizado correctamente.")

def test_create_cnn_model_missing_num_channels():
    """
    Test que envía un mensaje sin la clave 'num_channels' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("num_channels")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'num_channels' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_px_h():
    """
    Test que envía un mensaje sin la clave 'px_h' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("px_h")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'px_h' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_px_w():
    """
    Test que envía un mensaje sin la clave 'px_w' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("px_w")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'px_w' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_num_classes():
    """
    Test que envía un mensaje sin la clave 'num_classes' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("num_classes")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'num_classes' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_Batch_Size():
    """
    Test que envía un mensaje sin la clave 'Batch Size' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("Batch Size")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'Batch Size' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_Optimizer():
    """
    Test que envía un mensaje sin la clave 'Optimizer' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("Optimizer")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'Optimizer' finalizado correctamente con status_code 400.")

def test_create_cnn_model_missing_Learning_Rate():
    """
    Test que envía un mensaje sin la clave 'Learning Rate' y verifica que se reciba un error (status_code 400).
    """
    params = get_valid_params()
    params.pop("Learning Rate")
    message = json.dumps(params)
    
    producer = create_producer()
    send_message(producer, TOPIC, key="cnn_model", value=message)
    
    consumer = create_consumer()
    response = consume_message(consumer, max_wait=30)
    consumer.close()
    
    check_error_response(response)
    print("✅ Test sin 'Learning Rate' finalizado correctamente con status_code 400.")

if __name__ == "__main__":
    print("Ejecutando test de creación de CNN válido...")
    test_create_cnn_model_valid()
    print("\nEjecutando test sin 'num_channels'...")
    test_create_cnn_model_missing_num_channels()
    print("\nEjecutando test sin 'px_h'...")
    test_create_cnn_model_missing_px_h()
    print("\nEjecutando test sin 'px_w'...")
    test_create_cnn_model_missing_px_w()
    print("\nEjecutando test sin 'num_classes'...")
    test_create_cnn_model_missing_num_classes()
    print("\nEjecutando test sin 'Batch Size'...")
    test_create_cnn_model_missing_Batch_Size()
    print("\nEjecutando test sin 'Optimizer'...")
    test_create_cnn_model_missing_Optimizer()
    print("\nEjecutando test sin 'Learning Rate'...")
    test_create_cnn_model_missing_Learning_Rate()
