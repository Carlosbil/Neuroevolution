from utils import generate_random_model_config, check_initial_poblation, logger
from responses import ok_message, bad_model_message

def handle_create_initial_population(topic, params):
    """
    Procesa el mensaje del tópico create_initial_population.
    Genera una población inicial de modelos aleatorios y devuelve la respuesta.
    """
    logger.info(f"Procesando create_initial_population con parámetros: {params}")
    if not check_initial_poblation(params):
        return bad_model_message(topic, "Invalid initial population request, check request params")

    num_population = params.get('num_poblation', 10)
    models = {}
    for i in range(num_population):
        models[i] = generate_random_model_config(
            params.get('num_channels', 3),
            params.get('px_h', 32),
            params.get('px_w', 32),
            params.get('num_classes', 10),
            params.get('batch_size', 32)
        )

    response = {
        'message': 'CNN initial population created successfully',
        'models': models
    }
    return ok_message(topic, response)
