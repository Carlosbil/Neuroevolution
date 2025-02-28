from utils import get_individual, get_dataset_params, create_genome, cross_genomes, mutate_genome, logger
from responses import ok_message, bad_model_message

def handle_create_child(topic, params):
    """
    Procesa el mensaje del tópico create_child.
    Realiza el cruce y mutación de dos individuos y devuelve la respuesta.
    """
    logger.info(f"Procesando create_child con parámetros: {params}")
    try:
        individual = get_individual(params['1'])
        second_individual = get_individual(params['2'])
        dataset_params = get_dataset_params(params['1'])
        second_dataset_params = get_dataset_params(params['2'])
        gen_1 = create_genome(individual, dataset_params)
        gen_2 = create_genome(second_individual, second_dataset_params)
        crossed_individual = cross_genomes(genome1=gen_1, genome2=gen_2)
        mutated_individual = mutate_genome(crossed_individual)
        logger.info(f"Resultado: gen1={gen_1}, gen2={gen_2}, cruzado={crossed_individual}, mutado={mutated_individual}")
    except ValueError as e:
        return bad_model_message()

    response = {
        'message': 'CNN model crossed and mutated successfully',
        'model': mutated_individual,
    }
    return ok_message(response)
