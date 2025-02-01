# models.py
from flask_restx import fields, Api

api = Api()

layer_structure = api.model('LayerStructure', {
    'Number of Convolutional Layers': fields.Integer(required=True, description='Number of convolutional layers'),
    'Number of Fully Connected Layers': fields.Integer(required=True, description='Number of fully connected layers'),
    'Number of Nodes in Each Layer': fields.List(fields.Integer, required=True, description='Nodes per layer'),
    'Activation Functions': fields.List(fields.String, required=True, description='Activation functions'),
    'Dropout Rate': fields.Float(required=True, description='Dropout rate'),
    'Learning Rate': fields.Float(required=True, description='Learning rate'),
    'Batch Size': fields.Integer(required=True, description='Batch size'),
    'Optimizer': fields.String(required=True, description='Optimizer used'),
    'num_channels': fields.Integer(required=True, description='Number of input channels'),
    'px_h': fields.Integer(required=True, description='Image height'),
    'px_w': fields.Integer(required=True, description='Image width'),
    'num_classes': fields.Integer(required=True, description='Number of classes'),
    'filters': fields.List(fields.Integer, required=True, description='Filter sizes'),
    'kernel_sizes': fields.List(fields.Integer, required=True, description='Kernel sizes')
})

cnn_model_parameters = api.model('CNNModelParameters', {
    '1': fields.Nested(layer_structure, required=True, description='The first individual'),
    '2': fields.Nested(layer_structure, required=True, description='The second individual')
})