# Neuroevolution System

## 📋 Descripción General del Proyecto

**Neuroevolution** es un sistema distribuido avanzado para la evolución automática de arquitecturas de redes neuronales convolucionales (CNNs) utilizando algoritmos genéticos. El proyecto implementa una plataforma completa de neuroevolución que permite descubrir arquitecturas óptimas de deep learning sin intervención manual, utilizando principios de computación evolutiva.

### 🎯 Objetivo Principal

El sistema está diseñado específicamente para **detección de Parkinson mediante análisis de audio**, convirtiendo señales de audio en espectrogramas (series temporales representadas como imágenes) y utilizando CNNs evolucionadas genéticamente para clasificación. Sin embargo, la arquitectura es completamente genérica y puede adaptarse a cualquier problema de clasificación con imágenes.

### ⚡ Características Clave

- **Neuroevolución Automática**: Descubrimiento automático de arquitecturas CNN óptimas
- **Procesamiento de Series Temporales**: Conversión de audio a espectrogramas para análisis temporal con CNNs
- **Arquitectura Distribuida**: Sistema basado en microservicios con comunicación mediante Kafka
- **Algoritmos Genéticos Híbridos**: Implementación de Hybrid NEAT con criterios de convergencia inteligentes
- **Evaluación Paralela**: Entrenamiento y evaluación simultánea de múltiples arquitecturas
- **Detección de Parkinson**: Aplicación práctica en análisis de patrones vocales
- **Escalabilidad**: Diseño distribuido que permite procesamiento de grandes poblaciones

## 🔬 Contexto Científico: CNNs para Series Temporales

### ¿Por qué CNNs para Series Temporales de Audio?

Tradicionalmente, las **Redes Neuronales Recurrentes (RNNs)** se consideraban la opción natural para series temporales. Sin embargo, este proyecto demuestra que las **CNNs pueden ser superiores** cuando se utiliza la representación adecuada:

1. **Transformación Tiempo-Frecuencia**: El audio se convierte en espectrogramas, que son representaciones 2D donde:
   - **Eje X**: Tiempo
   - **Eje Y**: Frecuencia
   - **Intensidad**: Amplitud/potencia en escala de colores

2. **Ventajas de CNNs sobre RNNs para Audio**:
   - ✅ **Extracción de patrones espaciales**: Las CNNs detectan características locales en tiempo y frecuencia simultáneamente
   - ✅ **Paralelización**: Entrenamiento más rápido que RNNs secuenciales
   - ✅ **Invarianza traslacional**: Detecta patrones independientemente de su posición temporal
   - ✅ **Jerarquía de características**: Capas sucesivas aprenden desde texturas básicas hasta patrones complejos
   - ✅ **Menos problemas de gradiente**: No sufren vanishing/exploding gradients como las RNNs

3. **Aplicación a Parkinson**:
   - Los patrones vocales de pacientes con Parkinson muestran características distintivas en frecuencia y tiempo
   - Los espectrogramas revelan temblores vocales, variaciones de tono y otros biomarcadores
   - Las CNNs aprenden automáticamente estas características sin feature engineering manual

**📚 Para más detalles técnicos sobre el uso de CNNs para series temporales, consulta [`CNN_TIME_SERIES.md`](./CNN_TIME_SERIES.md)**

## 🏗️ Overview Técnico

Neuroevolution es un sistema distribuido que combina computación evolutiva con deep learning para evolucionar arquitecturas de redes neuronales. El sistema utiliza Apache Kafka para comunicación asíncrona entre componentes, permitiendo escalabilidad y tolerancia a fallos.

## 🏗️ Arquitectura del Sistema

El sistema está compuesto por cuatro microservicios principales que se comunican a través de Apache Kafka:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUJO GENERAL                             │
│                                                                  │
│  Cliente → Broker → GeneticAlgorithmService → Broker            │
│                          ↓                                       │
│                    Genome Service                                │
│                          ↓                                       │
│                    Evolutioners                                  │
│                          ↓                                       │
│                PostgreSQL (Resultados)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 🎯 Broker (Orquestador Central)

**Función**: Hub central de comunicación que coordina el flujo de mensajes entre componentes.

**Responsabilidades**:
- ✅ Crear poblaciones iniciales de arquitecturas CNN
- ✅ Gestionar evaluación de poblaciones
- ✅ Seleccionar mejores arquitecturas (elitismo)
- ✅ Coordinar creación de modelos hijo (crossover)
- ✅ Almacenar resultados en PostgreSQL
- ✅ Proporcionar endpoints de estado

**Tecnologías**: Python, Flask, Kafka Consumer/Producer, PostgreSQL

**Topics Kafka principales**:
- `create-initial-population` / `...-response`
- `evaluate-population` / `...-response`
- `select-best-architectures` / `...-response`
- `create-child` / `...-response`

### 2. 🧬 Genome Service (Operaciones Genéticas)

**Función**: Implementa operaciones genéticas sobre arquitecturas neuronales.

**Responsabilidades**:
- 🔄 Generar poblaciones iniciales aleatorias
- 🔀 **Crossover**: Combinar dos arquitecturas padre en un hijo
- 🎲 **Mutación**: Introducir variaciones aleatorias en arquitecturas
  - Cambiar número de capas
  - Modificar tamaños de filtros
  - Alterar funciones de activación
  - Ajustar tasa de dropout

**Parámetros evolucionables**:
```python
genome = {
    'Number of Convolutional Layers': [1-5],    # Profundidad red
    'filters': [16, 32, 64, 128, 256],          # Canales por capa
    'kernel_sizes': [(3,3), (5,5), (7,7)],      # Tamaño filtros
    'Activation Functions': ['relu', 'tanh', 'selu', 'leakyrelu'],
    'Number of Fully Connected Layers': [1-3],
    'Dropout Rate': [0.1 - 0.5],
    'Learning Rate': [0.0001 - 0.01],
    'Optimizer': ['adam', 'adamw', 'sgd', 'rmsprop']
}
```

**Topics Kafka**:
- `genome-create-initial-population`
- `genome-create-child`

### 3. ⚡ Evolutioners (Entrenamiento CNN)

**Función**: Construir, entrenar y evaluar arquitecturas CNN desde genomas.

**Responsabilidades**:
- 🏗️ Construir arquitectura PyTorch desde especificación genómica
- 📊 Cargar datasets (espectrogramas de Parkinson o MNIST)
- 🎓 Entrenar modelos (3 epochs por defecto)
- 📈 Evaluar fitness (accuracy en test set)
- 🔙 Reportar resultados al Broker

**Pipeline de procesamiento**:
```python
1. Recibir genoma: {conv_layers, filters, activations, ...}
2. Construir CNN: Sequential(Conv2d → BatchNorm → ReLU → MaxPool, ...)
3. Cargar espectrogramas: ImageFolder(path='images_parkinson/')
4. Entrenar: 3 epochs con optimizer especificado
5. Evaluar: Accuracy en test set
6. Retornar: {"score": 94.2, "model_id": "...", "uuid": "..."}
```

**Optimizaciones**:
- ⚡ Aceleración GPU (CUDA)
- 🔄 Workers paralelos para carga de datos
- 💾 Pin memory para transferencias rápidas
- 🧹 Liberación de caché CUDA entre modelos

**Topics Kafka**:
- `evolutioner-create-cnn-model`

### 4. 🧠 GeneticAlgorithmService (Motor Evolutivo)

**Función**: Orquestar el algoritmo genético completo con múltiples generaciones.

**Responsabilidades**:
- 🔁 Gestionar ciclos evolutivos multi-generacionales
- 📊 Implementar criterios de convergencia
- 🛑 Detección automática de parada:
  - Máximo de generaciones alcanzado
  - Umbral de fitness objetivo logrado
  - Estancamiento detectado (fitness no mejora)
- 📈 Tracking de progreso evolutivo
- 🏆 Selección del mejor modelo global

**Flujo del algoritmo genético**:
```
Inicialización:
├─ Crear población inicial (N individuos aleatorios)
└─ Evaluar fitness de cada individuo

Para cada Generación (hasta max_generations):
├─ Evaluar población actual
├─ Verificar criterios de convergencia
│  ├─ ✅ Si fitness >= threshold → DETENER
│  ├─ ✅ Si generación >= max_gen → DETENER
│  └─ ✅ Si estancamiento → DETENER
├─ Seleccionar mejores (elitismo 50%)
├─ Crear hijos mediante crossover + mutación
├─ Reemplazar población con nueva generación
└─ Actualizar mejor global

Finalización:
└─ Retornar mejor arquitectura + estadísticas
```

**Criterios de convergencia**:
```python
def check_convergence(gen, max_gen, best_fit, threshold, history):
    # 1. Máximo de generaciones
    if gen >= max_gen:
        return True, "Max generations reached"
    
    # 2. Umbral de fitness
    if best_fit >= threshold:
        return True, f"Fitness threshold {threshold}% achieved"
    
    # 3. Estancamiento (no mejora en 5 generaciones)
    if len(history) >= 5 and all(abs(history[-1] - h) < 0.01 for h in history[-5:]):
        return True, "Fitness stagnation detected"
    
    return False, "Continue evolution"
```

**Topics Kafka**:
- `genetic-algorithm`
- `continue-algorithm`

### 5. 💾 PostgreSQL (Persistencia)

**Función**: Almacenar resultados de evolución y metadatos de poblaciones.

**Esquema principal**:
```sql
CREATE TABLE populations (
    uuid VARCHAR PRIMARY KEY,
    generation INT,
    population_data JSONB,  -- Genomas de individuos
    fitness_scores JSONB,   -- Accuracy de cada individuo
    best_fitness FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE evaluations (
    uuid VARCHAR,
    model_id VARCHAR,
    accuracy FLOAT,
    architecture JSONB,
    training_time FLOAT,
    evaluated_at TIMESTAMP
);
```

**Queries importantes**:
- Obtener mejor arquitectura histórica
- Tracking de evolución de fitness
- Análisis de convergencia
- Comparación entre generaciones

---

## 🔄 Communication Flow (Flujo Completo)

### Diagrama de Secuencia Detallado

```
Usuario                    Broker              GeneticAlgorithm    Genome          Evolutioners      PostgreSQL
  │                          │                        │              │                   │               │
  │──genetic-algorithm───────>│                        │              │                   │               │
  │  {num_pop: 20,            │                        │              │                   │               │
  │   max_gen: 50}            │                        │              │                   │               │
  │                           │                        │              │                   │               │
  │                           │──genetic-algorithm─────>│              │                   │               │
  │                           │                        │              │                   │               │
  │                           │                        │──create-pop──>│                   │               │
  │                           │                        │              │                   │               │
  │                           │                        │<──20 genomas─┤                   │               │
  │                           │                        │              │                   │               │
  │                           │                        │──────────────────evaluate-pop────>│               │
  │                           │                        │              │   (20 modelos)    │               │
  │                           │                        │              │                   │               │
  │                           │                        │              │    [Construcción] │               │
  │                           │                        │              │    [Entrenamiento]│               │
  │                           │                        │              │    [Evaluación]   │               │
  │                           │                        │              │                   │               │
  │                           │                        │<─────────────────fitness scores──┤               │
  │                           │                        │              │   [78%, 82%, ...] │               │
  │                           │                        │                                  │               │
  │                           │                        │──────────────────────────────────────save────────>│
  │                           │                        │                                  │               │
  │                           │                        │──select-best─>│                  │               │
  │                           │                        │  (top 50%)    │                  │               │
  │                           │                        │              │                   │               │
  │                           │                        │──create-child->│                  │               │
  │                           │                        │  (crossover)  │                  │               │
  │                           │                        │              │                   │               │
  │                           │                        │<──new genomas─┤                  │               │
  │                           │                        │              │                   │               │
  │                           │                        │──[LOOP: Gen 2-50]────────────────>│               │
  │                           │                        │                                                  │
  │                           │                        │──[CONVERGENCE CHECK]                             │
  │                           │                        │    gen=15, fitness=94.2%                         │
  │                           │                        │    threshold=95% → CONTINUE                      │
  │                           │                        │                                                  │
  │                           │                        │    gen=23, fitness=95.3%                         │
  │                           │                        │    threshold=95% → STOP! ✅                       │
  │                           │                        │                                                  │
  │                           │<──response────────────┤                                                  │
  │<──response────────────────┤   {best_uuid: "...",                                                     │
  │   {best_model: "...",     │    fitness: 95.3%,                                                       │
  │    generations: 23,       │    reason: "threshold"}                                                  │
  │    fitness: 95.3%}        │                                                                          │
```

### Ejemplo de Mensaje Kafka

**Topic**: `genetic-algorithm`
```json
{
  "num_channels": 1,
  "px_h": 128,
  "px_w": 128,
  "num_classes": 2,
  "batch_size": 32,
  "num_poblation": 20,
  "max_generations": 50,
  "fitness_threshold": 95.0,
  "mutation_rate": 0.1,
  "path": "/data/parkinson_spectrograms"
}
```

**Response**: `genetic-algorithm-response`
```json
{
  "status": "success",
  "uuid": "abc123-best-model",
  "generation": 23,
  "converged": true,
  "convergence_reason": "Fitness threshold 95.0% achieved",
  "best_fitness": 95.3,
  "fitness_history": [65.3, 72.1, 78.4, ..., 94.8, 95.3],
  "best_architecture": {
    "num_conv_layers": 4,
    "filters": [32, 64, 128, 256],
    "kernel_sizes": [[3,3], [3,3], [3,3], [3,3]],
    "activation": ["relu", "relu", "relu", "relu"],
    "dropout": 0.35,
    "fully_connected": 2
  },
  "training_time_total": "2h 15m",
  "models_evaluated": 460
}
```

## Communication Flow

The components communicate through Kafka topics:

```
Client → Broker → GeneticAlgorithmService → Broker → Genome/Evolutioners
```

## 📦 Instalación y Configuración

### Prerequisitos

- **Python 3.8+** (recomendado 3.10)
- **CUDA 11.0+** (opcional, para aceleración GPU)
- **Docker & Docker Compose** (opcional, para despliegue containerizado)
- **8GB RAM mínimo** (16GB recomendado para poblaciones grandes)
- **Espacio en disco**: ~5GB para dependencias + dataset

### 🚀 Setup Rápido con Docker (Recomendado)

El método más sencillo para iniciar todo el stack:

```bash
# 1. Clonar repositorio
git clone https://github.com/Carlosbil/Neuroevolution.git
cd Neuroevolution

# 2. Configurar variables de entorno (opcional)
cp .env.example .env
# Editar .env con tus configuraciones

# 3. Iniciar servicios con Docker Compose
docker-compose up -d

# 4. Verificar que todos los servicios están corriendo
docker-compose ps
# Deberías ver: zookeeper, kafka, postgres, genome (y opcionalmente broker, evolutioners)

# 5. Ver logs en tiempo real
docker-compose logs -f
```

**Servicios iniciados**:
- ✅ Zookeeper (coordinación Kafka) - Puerto 2181
- ✅ Kafka (message broker) - Puerto 9092
- ✅ PostgreSQL (persistencia) - Puerto 5432
- ✅ Genome Service (operaciones genéticas)
- ✅ Topics Kafka creados automáticamente

### 🛠️ Setup Manual (Desarrollo Local)

Para desarrollo o si prefieres control granular:

#### 1. Clonar e Instalar Dependencias

```bash
# Clonar repositorio
git clone https://github.com/Carlosbil/Neuroevolution.git
cd Neuroevolution

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias por servicio (opcional)
pip install -r Broker/requirements.txt
pip install -r Evolutioners/requirements.txt
pip install -r Genome/requirements.txt
pip install -r GeneticAlgorithmService/requirements.txt
```

#### 2. Iniciar Kafka y PostgreSQL

**Opción A: Usar Docker solo para infraestructura**
```bash
# Iniciar solo Kafka y PostgreSQL
docker-compose up -d zookeeper kafka postgres kafka-init
```

**Opción B: Kafka local (manual)**
```bash
# Descargar Kafka
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Iniciar Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties &

# Iniciar Kafka
bin/kafka-server-start.sh config/server.properties &

# Crear topics necesarios
bin/kafka-topics.sh --create --topic genetic-algorithm --bootstrap-server localhost:9092
bin/kafka-topics.sh --create --topic create-initial-population --bootstrap-server localhost:9092
# ... (crear todos los topics listados en docker-compose.yaml)
```

**Opción C: PostgreSQL local**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# MacOS (Homebrew)
brew install postgresql
brew services start postgresql

# Crear base de datos
psql -U postgres
CREATE DATABASE neat_db;
CREATE USER neat_user WITH PASSWORD 'neat_pass';
GRANT ALL PRIVILEGES ON DATABASE neat_db TO neat_user;
```

#### 3. Configurar Variables de Entorno

```bash
# Crear archivo .env en raíz del proyecto
cat > .env << EOF
KAFKA_BROKER=localhost:9092
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=neat_user
POSTGRES_PASSWORD=neat_pass
POSTGRES_DB=neat_db
EOF
```

#### 4. Iniciar Servicios Manualmente

Abrir **4 terminales separadas**:

**Terminal 1: Broker**
```bash
cd Broker
export KAFKA_BROKER=localhost:9092
export POSTGRES_HOST=localhost
python app.py
# Output: Broker listening on Kafka topics...
```

**Terminal 2: Genome Service**
```bash
cd Genome
export KAFKA_BROKER=localhost:9092
python app.py
# Output: Genome service ready for genetic operations...
```

**Terminal 3: Evolutioners**
```bash
cd Evolutioners
export KAFKA_BROKER=localhost:9092
python app.py
# Output: Evolutioners ready to train CNNs...
```

**Terminal 4: GeneticAlgorithmService**
```bash
cd GeneticAlgorithmService
export KAFKA_BROKER=localhost:9092
python app.py
# Output: Genetic Algorithm Service ready...
```

### 🧪 Verificar Instalación

```bash
# Verificar conectividad a Kafka
python -c "from confluent_kafka import Producer; p = Producer({'bootstrap.servers': 'localhost:9092'}); print('✅ Kafka OK')"

# Verificar PyTorch y CUDA (si GPU disponible)
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA disponible: {torch.cuda.is_available()}')"

# Verificar PostgreSQL
python -c "import psycopg2; conn = psycopg2.connect(host='localhost', database='neat_db', user='neat_user', password='neat_pass'); print('✅ PostgreSQL OK'); conn.close()"

# Verificar servicios con health check
curl http://localhost:5000/health  # Si Broker tiene endpoint de salud
```

### 📊 Preparar Dataset de Parkinson (Opcional)

Si deseas usar el proyecto para detección de Parkinson:

```bash
# 1. Organizar archivos de audio
mkdir -p data/parkinson_audio
# Estructura:
# data/parkinson_audio/
#   ├── pretrained_control/           # Audio de pacientes sanos
#   │   └── pretrained_control/       # Subdirectorio con .wav files
#   └── pretrained_pathological/      # Audio de pacientes Parkinson
#       └── pretrained_pathological/  # Subdirectorio con .wav files

# 2. Convertir audio a espectrogramas
python wav_to_images_converter.py
# Seguir instrucciones interactivas:
#   - Ruta base: ./data/parkinson_audio
#   - Usar GPU: S
#   - Tipo: 1 (Solo espectrogramas)

# 3. Resultado:
# data/parkinson_audio/
#   ├── images_control/           # Espectrogramas de sanos (PNG)
#   └── images_pathological/      # Espectrogramas de Parkinson (PNG)
```

### 🎓 Dataset de Prueba (MNIST)

Si solo quieres probar el sistema sin datos de Parkinson:

```python
# El sistema descargará automáticamente MNIST la primera vez
# No requiere preparación manual
# Ver "Usage" sección para ejemplo con MNIST
```

## 🎮 Uso del Sistema

### 📝 Inicio Rápido: Ejemplo Completo

#### Opción 1: Script de Lanzamiento (Más Fácil)

```bash
# 1. Asegurarse de que todos los servicios están corriendo
docker-compose ps  # o verifica manualmente cada servicio

# 2. Ejecutar algoritmo genético con configuración por defecto
cd Broker/flows
python start_genetic_algorithm.py

# Output esperado:
# 🚀 Iniciando algoritmo genético controlado
# 📋 Configuración: {...}
# 🛑 Criterios de parada configurados:
#    - Máximo 5 generaciones
#    - Fitness objetivo: 99%
# ✅ Mensaje enviado al tópico 'genetic-algorithm'
# 💡 El algoritmo se detendrá automáticamente cuando:
#    • Se alcance el fitness objetivo
#    • Se complete el número máximo de generaciones
#    • Se detecte estancamiento en el fitness
```

#### Opción 2: Código Python Personalizado

```python
# lanzar_neuroevolucion.py
import json
from confluent_kafka import Producer

# Configurar productor Kafka
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Configuración del algoritmo genético
config = {
    # DATASET CONFIGURATION
    'num_channels': 1,           # 1=Grayscale, 3=RGB
    'px_h': 128,                 # Alto de espectrograma
    'px_w': 128,                 # Ancho de espectrograma
    'num_classes': 2,            # Sano vs Parkinson
    'path': './data/parkinson_audio/images_combined',  # Ruta a espectrogramas
    
    # TRAINING CONFIGURATION
    'batch_size': 32,            # Tamaño de batch (más grande = más rápido, más memoria)
    
    # GENETIC ALGORITHM CONFIGURATION
    'num_poblation': 20,         # Individuos por generación (más = mejor exploración)
    'max_generations': 50,       # Máximo de generaciones (criterio de parada)
    'fitness_threshold': 95.0,   # Detener si accuracy >= 95% (criterio de parada)
    'mutation_rate': 0.1,        # Probabilidad de mutación (0.1 = 10%)
}

# Enviar mensaje a Kafka
message = json.dumps(config)
producer.produce('genetic-algorithm', message.encode('utf-8'))
producer.flush()

print("✅ Algoritmo genético iniciado!")
print("📊 Monitorea el progreso en los logs de los servicios")
```

```bash
# Ejecutar
python lanzar_neuroevolucion.py
```

### 📊 Casos de Uso Comunes

#### Caso 1: Detección de Parkinson (Dataset Personalizado)

```python
# 1. Preparar datos
python wav_to_images_converter.py
# Seleccionar: ruta_base = './data/parkinson_audio'
#              usar_gpu = Sí
#              tipo = Solo espectrogramas

# 2. Configurar y ejecutar
config_parkinson = {
    'num_channels': 1,        # Espectrogramas son grayscale
    'px_h': 128,
    'px_w': 128,
    'num_classes': 2,         # Binario: Sano vs Parkinson
    'batch_size': 32,
    'num_poblation': 20,      # 20 arquitecturas diferentes
    'max_generations': 50,
    'fitness_threshold': 95.0,
    'mutation_rate': 0.1,
    'path': './data/parkinson_audio/images_combined'  # Espectrogramas
}

# Enviar a Kafka
producer.produce('genetic-algorithm', json.dumps(config_parkinson))
producer.flush()
```

**Resultados esperados**:
- Generación 0: Accuracy ~65% (arquitecturas aleatorias)
- Generación 10: Accuracy ~82% (primeras optimizaciones)
- Generación 25: Accuracy ~92% (convergencia)
- Generación 35: Accuracy ~95% ✅ (threshold alcanzado, detención automática)
- Tiempo total: ~2-3 horas (GPU) o ~8-12 horas (CPU)

#### Caso 2: Clasificación MNIST (Dataset de Prueba)

```python
config_mnist = {
    'num_channels': 1,        # Imágenes grayscale
    'px_h': 28,
    'px_w': 28,
    'num_classes': 10,        # Dígitos 0-9
    'batch_size': 128,        # Batch más grande (dataset simple)
    'num_poblation': 15,
    'max_generations': 30,
    'fitness_threshold': 99.0,  # MNIST es más fácil, threshold más alto
    'mutation_rate': 0.15,
    # NO incluir 'path' → sistema usa MNIST automáticamente
}

producer.produce('genetic-algorithm', json.dumps(config_mnist))
producer.flush()
```

**Ventajas MNIST**:
- ✅ No requiere preparación de datos
- ✅ Descarga automática la primera vez
- ✅ Rápido para pruebas (~30 min)
- ✅ Ideal para validar que el sistema funciona

#### Caso 3: Dataset Personalizado (Cualquier Clasificación)

```python
# Estructura requerida:
# mi_dataset/
#   ├── clase_A/
#   │   ├── imagen_001.png
#   │   ├── imagen_002.png
#   │   └── ...
#   ├── clase_B/
#   │   └── ...
#   └── clase_C/
#       └── ...

config_custom = {
    'num_channels': 3,        # RGB si es color, 1 si es grayscale
    'px_h': 224,              # Ajustar según tus imágenes
    'px_w': 224,
    'num_classes': 3,         # Número de carpetas (clases)
    'batch_size': 16,         # Más bajo si imágenes grandes
    'num_poblation': 10,
    'max_generations': 40,
    'fitness_threshold': 90.0,
    'mutation_rate': 0.12,
    'path': '/ruta/absoluta/a/mi_dataset'  # IMPORTANTE: ruta absoluta
}
```

### 🔍 Monitoreo de Progreso

#### Ver Logs en Tiempo Real

```bash
# Docker
docker-compose logs -f genetic-algorithm
docker-compose logs -f evolutioners
docker-compose logs -f broker

# Manual
# Logs aparecen en las terminales donde iniciaste cada servicio
```

**Ejemplo de logs del algoritmo genético**:
```
2025-10-10 12:00:00 - INFO - 🧬 Starting genetic algorithm - First generation
2025-10-10 12:00:05 - INFO - 🎲 Creating initial population of 20 individuals
2025-10-10 12:00:10 - INFO - ✅ Initial population created with UUID: abc-123
2025-10-10 12:05:30 - INFO - 📊 Generation 0 complete - Best fitness: 65.3%
2025-10-10 12:05:31 - INFO - 🔄 Creating Generation 1 (crossover + mutation)
2025-10-10 12:10:45 - INFO - 📊 Generation 1 complete - Best fitness: 72.1%
2025-10-10 12:10:46 - INFO - 🔄 Creating Generation 2
...
2025-10-10 14:30:00 - INFO - 📊 Generation 23 complete - Best fitness: 95.3%
2025-10-10 14:30:01 - INFO - ✅ Convergence achieved: Fitness threshold 95.0% reached!
2025-10-10 14:30:02 - INFO - 🏆 Best model UUID: abc-123-gen23-model5
```

#### Consultar Base de Datos

```bash
# Conectar a PostgreSQL
docker exec -it postgres psql -U neat_user -d neat_db

# O localmente
psql -U neat_user -d neat_db
```

```sql
-- Ver todas las poblaciones
SELECT uuid, generation, best_fitness, created_at 
FROM populations 
ORDER BY created_at DESC 
LIMIT 10;

-- Ver evolución de fitness por generación
SELECT generation, best_fitness 
FROM populations 
WHERE uuid LIKE 'abc-123%' 
ORDER BY generation;

-- Ver mejor modelo histórico
SELECT uuid, generation, best_fitness 
FROM populations 
ORDER BY best_fitness DESC 
LIMIT 1;

-- Ver detalles de arquitectura ganadora
SELECT population_data 
FROM populations 
WHERE uuid = 'abc-123-gen23-model5';
```

### 🎛️ Parámetros Avanzados

#### Ajuste de Hiperparámetros del Algoritmo Genético

```python
config_advanced = {
    # Dataset básico
    'num_channels': 1,
    'px_h': 128,
    'px_w': 128,
    'num_classes': 2,
    'path': './data/espectrogramas',
    
    # Hiperparámetros de entrenamiento
    'batch_size': 64,              # ↑ Mayor = más rápido pero más memoria
    
    # Hiperparámetros evolutivos
    'num_poblation': 30,           # ↑ Mayor = mejor exploración pero más lento
    'max_generations': 100,        # ↑ Mayor = más tiempo de evolución
    'fitness_threshold': 97.0,     # ↑ Mayor = modelo más preciso pero más tiempo
    'mutation_rate': 0.15,         # ↑ Mayor = más diversidad pero menos convergencia
    
    # Opcionales (si están implementados)
    'elitism_rate': 0.2,           # % de mejores que pasan sin cambios (default: 0.5)
    'crossover_rate': 0.8,         # Probabilidad de crossover (default: 1.0)
    'tournament_size': 3,          # Tamaño torneo para selección (default: 2)
}
```

**Guía de ajuste**:

| Situación | Recomendación |
|-----------|---------------|
| Dataset pequeño (<500 muestras) | `num_poblation: 10-15`, `max_generations: 30-50` |
| Dataset grande (>5000 muestras) | `num_poblation: 30-50`, `batch_size: 64-128` |
| Problema complejo (muchas clases) | `max_generations: 80-150`, `mutation_rate: 0.2` |
| Tiempo limitado | `num_poblation: 10`, `max_generations: 20`, `fitness_threshold: 85` |
| Máxima precisión | `num_poblation: 50`, `max_generations: 200`, `fitness_threshold: 98` |
| Sin GPU | `batch_size: 16`, `num_poblation: 5` (muy lento) |

### 🔧 Depuración y Solución de Problemas

#### Verificar Conectividad

```python
# test_conexion.py
from confluent_kafka import Producer, Consumer

# Test Producer
try:
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    print("✅ Producer OK")
except Exception as e:
    print(f"❌ Producer Error: {e}")

# Test Consumer
try:
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'test-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['genetic-algorithm'])
    print("✅ Consumer OK")
except Exception as e:
    print(f"❌ Consumer Error: {e}")
```

#### Logs de Depuración

```bash
# Aumentar nivel de log
export LOG_LEVEL=DEBUG
python app.py  # En cada servicio
```

#### Verificar Topics Kafka

```bash
# Listar topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

# Ver mensajes en un topic (para debugging)
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic genetic-algorithm \
  --from-beginning
```

## Hybrid NEAT Philosophy

The system incorporates **Hybrid NEAT** (NeuroEvolution of Augmenting Topologies) principles to maximize each individual's potential through adaptive learning capabilities. The philosophy behind this approach is based on the principle that:

> **"The capacity to adapt and learn is superior to having a good foundation but becoming stagnant"**

### Why Hybrid NEAT?

Traditional genetic algorithms often focus on finding good initial architectures but may suffer from premature convergence or stagnation. Hybrid NEAT addresses this by:

1. **Adaptive Architecture Evolution**: Networks can dynamically adjust their topology during evolution, allowing them to discover optimal structures that weren't present in the initial population.

2. **Continuous Learning Capability**: Rather than relying solely on inherited "good genes," each individual maintains the ability to adapt and improve through structural modifications.

3. **Avoiding Stagnation**: The system prevents getting stuck in local optima by continuously introducing topological innovations and mutations that expand the search space.

4. **Maximizing Individual Potential**: Each network architecture is given the opportunity to reach its maximum potential through both genetic evolution and structural adaptation.

This approach ensures that the evolutionary process remains dynamic and exploratory, preventing the common problem where evolution converges too quickly to suboptimal solutions simply because they were "good enough" initially.

### Practical Benefits of Hybrid NEAT

This philosophy translates into several practical advantages:

1. **Higher Final Performance**: Networks that continue to adapt often achieve better final performance than those that start with good architectures but can't improve further.

2. **Robustness to Initial Conditions**: The system doesn't depend on having perfect initial architectures, making it more reliable across different problem domains.

3. **Discovery of Novel Solutions**: By maintaining adaptability, the system can discover unconventional but effective architectures that wouldn't emerge from traditional approaches.

4. **Scalability**: The adaptive nature allows the system to handle problems of varying complexity without requiring manual architecture design.

5. **Reduced Human Intervention**: The emphasis on adaptation reduces the need for domain expertise in neural architecture design.

> **Key Insight**: In neuroevolution, a network's ability to learn and adapt is more valuable than its initial performance. This is why Hybrid NEAT focuses on maintaining evolutionary pressure and diversity rather than just selecting the current best performers.

### Implementation in the System

The Hybrid NEAT approach is implemented through several key mechanisms:

- **Multi-generational Evolution**: The genetic algorithm runs for multiple generations, allowing architectures to continuously evolve and improve rather than stopping after a single evaluation.

- **Adaptive Mutation Rates**: The system uses intelligent mutation strategies that can modify network topology, layer sizes, activation functions, and other architectural parameters.

- **Fitness-based Selection with Diversity Preservation**: While selecting the best performers, the system maintains genetic diversity to prevent premature convergence to suboptimal solutions.

- **Incremental Complexity Growth**: Networks can start simple and gradually increase in complexity as needed, rather than beginning with overly complex architectures that may be hard to optimize.

The result is an evolutionary system that prioritizes **adaptability over initial perfection**, ensuring that each generation has the potential to surpass its predecessors through intelligent adaptation rather than just inheriting static "good" characteristics.

### Evolution Parameters

| Parameter | Description | MNIST | NEAT Impact |
|-----------|-------------|---------|-------------|
| `max_generations` | Maximum number of generations to evolve | 10 | 🔄 Allows continuous adaptation |
| `fitness_threshold` | Target fitness score (0.0-1.0) | 0.95 | 🎯 Prevents premature convergence |
| `num_poblation` | Population size | 10 | 🧬 Maintains genetic diversity |
| `num_channels` | Input channels (1=grayscale, 3=RGB) | 1 | 📊 Network input adaptation |
| `px_h`, `px_w` | Image dimensions | 28, 28 | 🖼️ Topology scaling capability |
| `num_classes` | Number of output classes | 10 | 🎯 Output layer adaptation |
| `batch_size` | Training batch size | 32 | ⚡ Training efficiency balance |

### Hybrid NEAT Specific Parameters

The system also supports additional parameters that leverage the Hybrid NEAT approach:

| Parameter | Description | Default | Purpose | Used on this proyect |
|-----------|-------------|---------|---------|---------|
| `mutation_rate` | Probability of genetic mutation | 0.1 | 🔄 Ensures continuous adaptation | Yes |

### Evolution Results

The genetic algorithm returns:
- **Best model UUID** and path
- **Generations completed**
- **Best fitness achieved**
- **Fitness history** across generations
- **Convergence reason** (threshold reached, max generations, stagnation)

## API Reference

### Kafka Topics

#### Input Topics

- `create-initial-population`: Create initial population of neural networks
- `evaluate-population`: Evaluate a population of neural networks
- `select-best-architectures`: Select the best architectures from a population
- `create-child`: Create a child model from two parent models
- `genetic-algorithm`: Run the complete genetic algorithm workflow

#### Response Topics

Each input topic has a corresponding response topic with the suffix `-response`.

### Message Formats

#### Create Initial Population

```json
{
  "num_channels": 1,
  "px_h": 28,
  "px_w": 28,
  "num_classes": 10,
  "batch_size": 32,
  "num_poblation": 10
}
```

#### Evaluate Population

```json
{
  "uuid": "population-uuid"
}
```

#### Create Child

```json
{
  "model_id": "parent1-id",
  "second_model_id": "parent2-id",
  "uuid": "population-uuid"
}
```

## Testing

The project includes integration tests to verify the functionality of each component:

```bash
# Run Broker integration tests
cd Broker
python integration_tests.py

# Run Evolutioners integration tests
cd Evolutioners
python integration_tests.py

# Run Genome integration tests
cd Genome
python integration_tests.py
```

## License

This project is licensed under the terms of the license included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🚀 Scripts de Lanzamiento

Se han creado varios scripts para facilitar el inicio del flujo de neuroevolución:

### Scripts Disponibles

1. **`start_flow.py`** - Script básico para lanzar el flujo con configuración por defecto
2. **`launch_neuroevolution_flow.py`** - Script completo con monitoreo de base de datos
3. **`check_database_status.py`** - Script para verificar el estado de la base de datos
4. **`docker_start_flow.py`** - Script optimizado para uso con Docker Compose

### Uso Rápido

```bash
# Lanzar flujo básico
python Broker/start_flow.py

# Lanzar con monitoreo completo
python Broker/launch_neuroevolution_flow.py

# Verificar estado de la base de datos
python Broker/check_database_status.py

# Para Docker Compose
python Broker/docker_start_flow.py
```

Consulta `Broker/LAUNCHER_README.md` para documentación detallada.