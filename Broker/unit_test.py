import json
import os
import pytest
import logging
from unittest import TestCase
from unittest.mock import patch, MagicMock
from topic_process.create_initial_population import process_create_initial_population
from utils import create_producer, produce_message, create_consumer 
from responses import ok_message, bad_model_message, response_message, runtime_error_message
from confluent_kafka import Producer, Consumer

# Configurar el logger para pytest
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCreateInitialPopulation(TestCase):
    def setUp(self):
        self.topic = "create_initial_population"
        self.valid_data = {
            "num_channels": 3,
            "px_h": 64,
            "px_w": 64,
            "num_classes": 10,
            "batch_size": 32,
            "num_poblation": 100
        }
        self.invalid_data = {"num_channels": 3}  # Datos incompletos
        self.mock_response = MagicMock()
    
    @patch("utils.Producer")
    def test_create_initial_population_success(self, mock_kafka_producer):
        mock_kafka_producer.return_value.produce.return_value = None
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"message": {"models": {"model_1": {}, "model_2": {}}}}

        with patch("app.requests.post", return_value=self.mock_response), \
             patch("app.generate_uuid", return_value="test_uuid"), \
             patch("builtins.open", create=True) as mock_open:
            
            result = process_create_initial_population(self.topic, self.valid_data)
            
            expected_result = ok_message(self.topic, {"uuid": "test_uuid", "path": os.path.join(os.path.dirname(__file__), 'models', 'test_uuid.json')})
            self.assertEqual(result, expected_result)
            mock_open.assert_called_once()
            logger.info("✅ test_create_initial_population_success PASSED")
    
    @patch("utils.Producer")
    def test_create_initial_population_bad_request(self, mock_kafka_producer):
        mock_kafka_producer.return_value.produce.return_value = None
        result = process_create_initial_population(self.topic, self.invalid_data)
        
        self.assertEqual(result, bad_model_message(self.topic))
        logger.info("✅ test_create_initial_population_bad_request PASSED")
    
    @patch("utils.Producer")
    def test_create_initial_population_server_error(self, mock_kafka_producer):
        mock_kafka_producer.return_value.produce.return_value = None
        self.mock_response.status_code = 500
        self.mock_response.json.return_value = {"error": "Internal Server Error"}

        with patch("app.requests.post", return_value=self.mock_response):
            result = process_create_initial_population(self.topic, self.valid_data)
            
            expected_result = response_message(self.topic, {"error": "Internal Server Error"}, 500)
            self.assertEqual(result, expected_result)
            logger.info("✅ test_create_initial_population_server_error PASSED")
    
    @patch("utils.Producer")
    def test_create_initial_population_runtime_error(self, mock_kafka_producer):
        mock_kafka_producer.return_value.produce.return_value = None
        with patch("app.requests.post", side_effect=Exception("Test Exception")):
            result = process_create_initial_population(self.topic, self.valid_data)
            
            self.assertEqual(result, runtime_error_message(self.topic))
            logger.info("✅ test_create_initial_population_runtime_error PASSED")

    def test_kafka_message_production(self):
        """Test que verifica que el mensaje se envía correctamente a Kafka."""
        producer_mock = MagicMock(spec=Producer)
        message = json.dumps(self.valid_data)
        
        with patch("utils.create_producer", return_value=producer_mock):
            produce_message(producer_mock, self.topic, message, times=1)
            producer_mock.produce.assert_called_with(self.topic, message.encode('utf-8'))
            producer_mock.flush.assert_called()
            logger.info("✅ test_kafka_message_production PASSED")