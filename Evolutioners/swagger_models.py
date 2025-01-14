# models.py
from flask_restx import fields, Api

api = Api()

cnn_model_parameters = api.model('CNNModelParameters', {
    'Number of Convolutional Layers': fields.Integer(required=True, description='Number of convolutional layers'),
    'Number of Fully Connected Layers': fields.Integer(required=True, description='Number of fully connected layers'),
    'Number of Nodes in Each Layer': fields.List(fields.Integer, required=True, description='List of nodes in each layer'),
    'Activation Functions': fields.List(fields.String, required=True, description='List of activation functions'),
    'Dropout Rate': fields.Float(required=True, description='Dropout rate'),
    'Learning Rate': fields.Float(required=True, description='Learning rate'),
    'Batch Size': fields.Integer(required=True, description='Batch size'),
    'Optimizer': fields.String(required=True, description='Optimizer (adam, sgd, rmsprop)'),
    'num_channels': fields.Integer(required=True, description='Number of channels'),
    'px_h': fields.Integer(required=True, description='Height of the input image'),
    'px_w': fields.Integer(required=True, description='Width of the input image'),
    'num_classes': fields.Integer(required=True, description='Number of classes'),
    'filters': fields.List(fields.Integer, required=True, description='List of filters'),
    'kernel_sizes': fields.List(fields.Integer, required=True, description='List of kernel sizes')
})