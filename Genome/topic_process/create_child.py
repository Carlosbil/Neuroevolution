from utils import get_individual, get_dataset_params, create_genome, cross_genomes, mutate_genome, logger
from responses import ok_message, bad_model_message, runtime_error_message, response_message
import random
def handle_create_child(topic, params):
    """
    Procesa el mensaje del tópico create_child.
    Realiza el cruce y mutación de dos individuos y devuelve la respuesta.
    """
    logger.info(f"Procesando create_child con parámetros: {params}")
    try:
        dataset_params = params.get("dataset_params", {})
        models = params.get("models", {})
        uuid = params.get("uuid", None)
        if dataset_params == {} or models == {}:
            return bad_request_message(topic,"Dataset params or models are empty")
        
        if uuid is None:
            return bad_request_message(topic,"UUID is empty")

        # Calculate number of children to generate (e.g., 50% of the current population)
        population_size = len(models)
        num_children = population_size
        children = {}
        for i in range(num_children):
            parent1 = random.choice(list(models.values()))
            parent2 = random.choice(list(models.values()))
            gen_1 = create_genome(parent1, dataset_params)
            gen_2 = create_genome(parent2, dataset_params)
            crossed_individual = cross_genomes(gen_1, gen_2)
            mutated_individual = mutate_genome(crossed_individual)
            children[str(i)] = mutated_individual



    except ValueError as e:
        return bad_model_message(topic)

    response = {
        'message': 'CNN model crossed and mutated successfully',
        'children': children,
        'uuid': uuid
    }
    return ok_message(topic, response)
