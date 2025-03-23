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

        childs = {}
        # generate the children crossing the models
        for i in models:
            second_model = random.randint(0, len(models))
            key, value = random.choice(list(models.items()))
            gen_1 = create_genome(models[i], dataset_params)
            gen_2 = create_genome(value, dataset_params)
            
            #cross both models
            crossed_individual = cross_genomes(genome1=gen_1, genome2=gen_2)
            
            #mutate both models
            mutated_individual = mutate_genome(crossed_individual)
            logger.info(f"Resultado: gen1={gen_1}, gen2={gen_2}, cruzado={crossed_individual}, mutado={mutated_individual}")
            
            # add child
            childs[i] = mutated_individual



    except ValueError as e:
        return bad_model_message(topic)

    response = {
        'message': 'CNN model crossed and mutated successfully',
        'models': childs,
        'uuid': uuid
    }
    return ok_message(topic, response)
