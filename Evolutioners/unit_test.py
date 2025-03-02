import pytest
from topic_process.create_cnn_model import handle_create_cnn_model

# Creamos un dummy para la función MNIST de torchvision para evitar descargas reales.
def dummy_MNIST(*args, **kwargs):
    # Retornamos una lista simple que actúa como dataset dummy
    return list(range(10))

def test_handle_create_cnn_model_success(monkeypatch):
    # Arrange
    topic = "test-topic"
    params = {"dummy": "value"}
    dummy_individual = "dummy_individual"
    dummy_dataset_params = {
        "batch_size": 2,
        "num_channels": 1,
        "px_h": 28,
        "px_w": 28,
        "num_classes": 10,
        "optimizer": "adam",
        "learning_rate": 0.001
    }
    dummy_accuracy = 0.95

    # Reemplazamos las funciones importadas en handler para no depender de implementaciones reales.
    import topic_process.create_cnn_model as handler
    monkeypatch.setattr(handler, "get_individual", lambda p: dummy_individual)
    monkeypatch.setattr(handler, "get_dataset_params", lambda p: dummy_dataset_params)
    monkeypatch.setattr(
        handler,
        "build_cnn_from_individual",
        lambda individual, num_channels, px_h, px_w, num_classes,
               train_loader, test_loader, optimizer_name, learning_rate, num_epochs: dummy_accuracy
    )
    monkeypatch.setattr(handler.datasets, "MNIST", dummy_MNIST)

    # Capturamos la llamada a ok_message
    captured = {}
    def fake_ok_message(t, message):
        captured['topic'] = t
        captured['message'] = message
    monkeypatch.setattr(handler, "ok_message", fake_ok_message)

    # Act: Ejecutamos la función
    handle_create_cnn_model(topic, params)

    # Assert: Verificamos que se llamó a ok_message con el mensaje esperado
    expected_response = {
        'message': 'CNN model created, trained and evaluated successfully',
        'score': dummy_accuracy,
    }
    assert captured.get('topic') == topic
    assert captured.get('message') == expected_response

def test_handle_create_cnn_model_value_error(monkeypatch):
    # Arrange
    topic = "test-topic"
    params = {"dummy": "value"}

    import topic_process.create_cnn_model as handler
    # Forzamos que get_individual lance un ValueError para simular error en la creación del modelo
    def fake_get_individual(p):
        raise ValueError("Invalid model")
    monkeypatch.setattr(handler, "get_individual", fake_get_individual)

    # Capturamos la llamada a bad_model_message
    captured = {}
    def fake_bad_model_message(t):
        captured['topic'] = t
    monkeypatch.setattr(handler, "bad_model_message", fake_bad_model_message)

    # Act: Ejecutamos la función
    handle_create_cnn_model(topic, params)

    # Assert: Verificamos que se llamó a bad_model_message con el topic correcto
    assert captured.get('topic') == topic
