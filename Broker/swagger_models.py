# models.py
from flask_restx import fields, Api

api = Api()

cnn_model_parameters = api.model('CNNModelParameters', {
    'model_id': fields.Integer(required=True, description='Model ID', default=1),
})

child_model_parameters = api.model('ChildModelParameters', {
    'model_id': fields.Integer(required=True, description='Model ID', default=1),
    'second_model_id': fields.Integer(required=True, description='Second Model ID', default=1),
})