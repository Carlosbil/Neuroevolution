# models.py
from flask_restx import fields, Api

api = Api()

cnn_model_parameters = api.model('CNNModelParameters', {
    'model_id': fields.Integer(required=True, description='Model ID', default=1),
    'uuid': fields.String(required=True, description='UUID', default=''),
})

child_model_parameters = api.model('ChildModelParameters', {
    'model_id': fields.Integer(required=True, description='Model ID', default=1),
    'second_model_id': fields.Integer(required=True, description='Second Model ID', default=1),
    'uuid': fields.String(required=True, description='UUID', default=''),
})

initial_poblation = api.model('InitialPoblation', {
    'num_channels': fields.Integer(required=True, description='Number of input channels', default=3),
    'px_h': fields.Integer(required=True, description='Image height', default=32),
    'px_w': fields.Integer(required=True, description='Image width', default=32),
    'num_classes': fields.Integer(required=True, description='Number of classes', default=10),
    'batch_size': fields.Integer(required=True, description='Batch size', default=32),
    'num_poblation': fields.Integer(required=True, description='Number of individuals in the initial poblation', default=10),
})