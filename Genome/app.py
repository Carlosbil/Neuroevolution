# app.py
from flask import Flask
from flask_restx import Api
from endpoints import ns as cnn_namespace
from swagger_models import api
import torch

app = Flask(__name__)

api.init_app(app)
api.add_namespace(cnn_namespace, path='/api')

if torch.cuda.is_available():
    print("CUDA (GPU) is available on your system.")
else:
    print("CUDA (GPU) is not available on your system.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
