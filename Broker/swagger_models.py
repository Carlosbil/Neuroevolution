# models.py
from flask_restx import fields, Api

api = Api()

cnn_model_parameters = api.model('CNNModelParameters', {
    'randomize': fields.Boolean(required=True, description='Randomize the model'),
})