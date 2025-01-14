import logging
import colorlog

logging.basicConfig(level=logging.DEBUG,  # Mínimo nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato del log
                    handlers=[logging.StreamHandler()])  # Mostrar los logs en consola
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Crear un handler para los logs en consola con colores
log_handler = colorlog.StreamHandler()

# Define el formato de los logs con colores
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',      # Debug será de color cian
        'INFO': 'green',      # Info será de color verde
        'WARNING': 'yellow',  # Warning será de color amarillo
        'ERROR': 'red',       # Error será de color rojo
        'CRITICAL': 'bold_red'  # Critical será de color rojo negrita
    }
)

# Establecer el formatter en el handler
log_handler.setFormatter(formatter)

# Añadir el handler al logger
logger.addHandler(log_handler)