# Lanzador del Flujo de Neuroevolución

Este directorio contiene scripts para iniciar el flujo de neuroevolución desde el principio y monitorear su progreso.

## Archivos

### `launch_neuroevolution_flow.py`
Script completo que:
- Lanza el tópico inicial de Kafka
- Monitorea la base de datos para verificar el progreso
- Proporciona información detallada sobre el estado del proceso
- Permite configurar parámetros personalizados

### `start_flow.py`
Script simple que solo lanza el tópico inicial con configuración por defecto para CIFAR-10.

### `check_models_directory.py`
Script para verificar y analizar el directorio de modelos, útil para debugging.

### `.env.example`
Archivo de configuración de ejemplo con las variables de entorno necesarias.

## Configuración

1. Copia el archivo `.env.example` a `.env` y ajusta los valores según tu configuración:
```bash
cp .env.example .env
```

2. Edita el archivo `.env` con tu configuración:
```bash
# Configuración de Kafka
KAFKA_BROKER=localhost:9092

# Configuración de PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=neuroevolution

# Configuración del Broker
BROKER_STORAGE_PATH=./models
```

## Uso

### Uso Básico (Recomendado para empezar)

```bash
# Lanzar el flujo con configuración por defecto
python start_flow.py
```

### Uso Avanzado con Monitoreo

```bash
# Lanzar con monitoreo completo y configuración por defecto
python launch_neuroevolution_flow.py

# Lanzar con parámetros personalizados
python launch_neuroevolution_flow.py \
    --num-channels 3 \
    --px-h 32 \
    --px-w 32 \
    --num-classes 10 \
    --batch-size 32 \
    --num-poblation 20 \
    --monitor-timeout 600 \
    --poll-interval 15
```

### Parámetros Disponibles

- `--num-channels`: Número de canales de entrada (default: 3)
- `--px-h`: Altura de las imágenes en píxeles (default: 32)
- `--px-w`: Ancho de las imágenes en píxeles (default: 32)
- `--num-classes`: Número de clases de salida (default: 10)
- `--batch-size`: Tamaño del batch para entrenamiento (default: 32)
- `--num-poblation`: Tamaño de la población inicial (default: 10)
- `--monitor-timeout`: Tiempo máximo de monitoreo en segundos (default: 300)
- `--poll-interval`: Intervalo entre verificaciones de la base de datos en segundos (default: 10)

## Ejemplos de Configuración

### CIFAR-10 (Por defecto)
```bash
python launch_neuroevolution_flow.py \
    --num-channels 3 \
    --px-h 32 \
    --px-w 32 \
    --num-classes 10 \
    --batch-size 32 \
    --num-poblation 10
```

### MNIST
```bash
python launch_neuroevolution_flow.py \
    --num-channels 1 \
    --px-h 28 \
    --px-w 28 \
    --num-classes 10 \
    --batch-size 64 \
    --num-poblation 15
```

### Población Grande con Monitoreo Extendido
```bash
python launch_neuroevolution_flow.py \
    --num-poblation 50 \
    --monitor-timeout 1800 \
    --poll-interval 30
```

## Monitoreo

El script `launch_neuroevolution_flow.py` proporciona información detallada sobre:

- Estado inicial de la base de datos
- Progreso en tiempo real
- Número de poblaciones y modelos creados
- Modelos con scores asignados
- Tiempo transcurrido y timeout restante

### Salida de Ejemplo

```
2025-01-11 10:00:00 - INFO - 🚀 Iniciando lanzador del flujo de neuroevolución
2025-01-11 10:00:00 - INFO - 📋 Configuración: {'num_channels': 3, 'px_h': 32, 'px_w': 32, 'num_classes': 10, 'batch_size': 32, 'num_poblation': 10}
2025-01-11 10:00:01 - INFO - 🔗 Conectando a la base de datos...
2025-01-11 10:00:01 - INFO - ✅ Conexión a la base de datos establecida
2025-01-11 10:00:01 - INFO - 📊 Verificando estado inicial de la base de datos...
2025-01-11 10:00:01 - INFO - 📈 Estado inicial - Poblaciones: 0, Modelos: 0
2025-01-11 10:00:01 - INFO - 🔗 Creando productor de Kafka...
2025-01-11 10:00:01 - INFO - ✅ Productor de Kafka creado
2025-01-11 10:00:01 - INFO - 🚀 Enviando mensaje inicial al tópico 'create-initial-population'
2025-01-11 10:00:01 - INFO - ✅ Mensaje enviado exitosamente al tópico 'create-initial-population'
2025-01-11 10:00:01 - INFO - 👁️ Iniciando monitoreo de la base de datos (timeout: 300s, intervalo: 10s)
2025-01-11 10:00:15 - INFO - 📊 Progreso detectado:
2025-01-11 10:00:15 - INFO -    - Poblaciones: 1 (+1)
2025-01-11 10:00:15 - INFO -    - Modelos: 10 (+10)
2025-01-11 10:00:15 - INFO -    - Modelos en última población: 10
2025-01-11 10:00:15 - INFO -    - Modelos con score: 0
2025-01-11 10:00:15 - INFO - 🎯 Objetivo alcanzado: 10/10 modelos
2025-01-11 10:00:15 - INFO - 🎉 Proceso completado exitosamente
```

## Requisitos

- Python 3.7+
- confluent-kafka
- psycopg2-binary
- python-dotenv
- colorlog

Instala las dependencias con:
```bash
pip install confluent-kafka psycopg2-binary python-dotenv colorlog
```

## Debug y Solución de Problemas

### Error "Invalid model"

Si recibes el error "Invalid model", sigue estos pasos:

1. **Verificar el directorio de modelos**:
```bash
python check_models_directory.py --list-all
```

2. **Verificar el UUID específico**:
```bash
python check_models_directory.py --analyze <uuid>
```

3. **Verificar los logs mejorados**:
Los logs ahora incluyen información detallada sobre:
- Ruta del archivo que se busca
- Archivos disponibles en el directorio
- Contenido del archivo JSON
- Detalles específicos del error

### Logs Mejorados

Los logs ahora proporcionan información mucho más detallada:

```
2025-01-11 10:00:01 - INFO - Processing evaluate_population with data: {'uuid': '9b535ea4-6c2f-497e-9476-c6666d5a6e0c'}
2025-01-11 10:00:01 - INFO - Looking for models file at path: /path/to/models/9b535ea4-6c2f-497e-9476-c6666d5a6e0c.json
2025-01-11 10:00:01 - ERROR - Models file not found at path: /path/to/models/9b535ea4-6c2f-497e-9476-c6666d5a6e0c.json
2025-01-11 10:00:01 - ERROR - Available files in storage directory: ['other-file.json', 'another-file.json']
```

### Crear Archivo de Prueba

Para probar el sistema, puedes crear un archivo de modelos de prueba:

```bash
python check_models_directory.py --create-test
```

Este comando creará un archivo JSON con modelos de ejemplo y te dará un UUID que puedes usar para pruebas.

## Solución de Problemas

### Error de Conexión a Kafka
- Verifica que Kafka esté ejecutándose
- Comprueba la configuración de `KAFKA_BROKER` en el archivo `.env`

### Error de Conexión a PostgreSQL
- Verifica que PostgreSQL esté ejecutándose
- Comprueba las configuraciones de conexión en el archivo `.env`
- Asegúrate de que la base de datos `neuroevolution` exista

### El Monitoreo No Detecta Progreso
- Verifica que el servicio Broker esté ejecutándose
- Comprueba los logs del Broker para ver si está procesando mensajes
- Verifica que los otros servicios (Genome, Evolutioners) estén ejecutándose

## Interrupción del Proceso

Puedes interrumpir el proceso de monitoreo en cualquier momento con `Ctrl+C`. El script realizará una limpieza ordenada de los recursos.

### Verificación del Directorio de Modelos

```bash
# Verificar el estado del directorio de modelos
python check_models_directory.py

# Listar y analizar todos los archivos JSON
python check_models_directory.py --list-all

# Analizar un archivo específico
python check_models_directory.py --analyze <uuid>

# Crear un archivo de prueba
python check_models_directory.py --create-test
```

## Ejemplos de Configuración
