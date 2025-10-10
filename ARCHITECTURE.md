# 🏗️ Arquitectura del Sistema Neuroevolution

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Componentes del Sistema](#componentes-del-sistema)
3. [Flujos de Datos](#flujos-de-datos)
4. [Diagramas de Arquitectura](#diagramas-de-arquitectura)
5. [Decisiones de Diseño](#decisiones-de-diseño)

---

## 🎯 Visión General

El sistema Neuroevolution está diseñado como una **arquitectura de microservicios distribuida** que utiliza **Apache Kafka** como bus de mensajería para comunicación asíncrona. Esta arquitectura permite:

- ✅ **Escalabilidad horizontal**: Añadir más instancias de cada servicio
- ✅ **Desacoplamiento**: Cada servicio opera independientemente
- ✅ **Tolerancia a fallos**: Si un servicio falla, los demás continúan
- ✅ **Procesamiento paralelo**: Múltiples modelos evaluándose simultáneamente
- ✅ **Persistencia**: PostgreSQL almacena resultados y estado

---

## 🧩 Componentes del Sistema

### Diagrama de Componentes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           SISTEMA NEUROEVOLUTION                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Usuario    │         │   Kafka      │         │  PostgreSQL  │
│   Cliente    │◄───────►│  Message Bus │◄───────►│   Database   │
└──────────────┘         └──────────────┘         └──────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐  ┌────▼─────┐  ┌──────▼──────┐
        │    Broker    │  │  Genome  │  │ Evolutioners│
        │ (Orquestador)│  │ (Genetic │  │  (Training) │
        └──────────────┘  │   Ops)   │  └─────────────┘
                          └──────────┘
                                │
                        ┌───────▼────────┐
                        │ Genetic Algo   │
                        │    Service     │
                        └────────────────┘
```

### 1. Kafka (Message Bus)

**Rol**: Sistema de mensajería distribuido que conecta todos los componentes.

**Topics principales**:
```
├── genetic-algorithm                      (Input: Iniciar evolución)
│   └── genetic-algorithm-response         (Output: Resultados finales)
├── create-initial-population              (Crear población inicial)
│   └── create-initial-population-response
├── evaluate-population                    (Evaluar fitness de población)
│   └── evaluate-population-response
├── genome-create-child                    (Crossover de genomas)
│   └── genome-create-child-response
├── evolutioner-create-cnn-model           (Entrenar modelo CNN)
│   └── evolutioner-create-cnn-model-response
└── select-best-architectures              (Seleccionar élite)
    └── select-best-architectures-response
```

**Características**:
- ⚡ Throughput: Millones de mensajes/segundo
- 🔄 Replicación: Garantiza no pérdida de mensajes
- ⏱️ Retención: Mensajes se mantienen configurablemente
- 📊 Particionamiento: Para procesamiento paralelo

### 2. Broker Service

**Responsabilidades**:
```python
class BrokerService:
    """
    Orquestador central del sistema
    """
    def handle_create_initial_population(self, params):
        """Delega a Genome Service para crear población"""
        
    def handle_evaluate_population(self, uuid):
        """Envía cada modelo a Evolutioners para entrenamiento"""
        
    def handle_select_best(self, uuid, fitness_scores):
        """Selecciona top 50% basado en fitness"""
        
    def handle_create_child(self, parent1, parent2):
        """Delega a Genome para crossover+mutación"""
        
    def save_to_database(self, population_data):
        """Persiste en PostgreSQL"""
```

**Interacciones**:
- 📨 Recibe jobs del usuario vía Kafka
- 🔀 Coordina flujo entre Genome y Evolutioners
- 💾 Almacena resultados en PostgreSQL
- 📊 Gestiona estado de poblaciones

### 3. Genome Service

**Responsabilidades**:
```python
class GenomeService:
    """
    Operaciones genéticas sobre arquitecturas neuronales
    """
    def create_initial_population(self, num_individuals):
        """
        Genera N genomas aleatorios
        
        Returns:
            [
                {
                    'num_conv_layers': random.randint(1, 5),
                    'filters': [random.choice([32, 64, 128, 256]) for _ in range(layers)],
                    'kernel_sizes': [random.choice([(3,3), (5,5)]) for _ in range(layers)],
                    'activation': [random.choice(['relu', 'tanh']) for _ in range(layers)],
                    'dropout': random.uniform(0.1, 0.5),
                    'fully_connected': random.randint(1, 3),
                    'learning_rate': random.uniform(0.0001, 0.01),
                    'optimizer': random.choice(['adam', 'adamw'])
                },
                ...
            ]
        """
    
    def crossover(self, parent1, parent2):
        """
        Combina dos genomas padre en un hijo
        
        Estrategias:
        - Uniform crossover: Cada gen tiene 50% probabilidad de venir de cualquier padre
        - Single-point crossover: Punto de corte aleatorio
        """
        
    def mutate(self, genome, mutation_rate=0.1):
        """
        Aplica mutaciones aleatorias
        
        Mutaciones posibles:
        - Añadir/eliminar capa convolucional
        - Cambiar número de filtros (±32)
        - Modificar función de activación
        - Ajustar dropout (±0.1)
        - Cambiar learning rate (*2 o /2)
        """
```

**Genoma Típico**:
```json
{
    "model_id": "gen5-ind3",
    "Number of Convolutional Layers": 4,
    "filters": [32, 64, 128, 256],
    "kernel_sizes": [[3,3], [3,3], [5,5], [3,3]],
    "Activation Functions": ["relu", "relu", "selu", "relu"],
    "Number of Fully Connected Layers": 2,
    "Number of Nodes in Each Layer": [512, 128],
    "Dropout Rate": 0.35,
    "Learning Rate": 0.001,
    "Optimizer": "adam",
    "Batch Size": 32
}
```

### 4. Evolutioners Service

**Responsabilidades**:
```python
class EvolutionersService:
    """
    Construcción, entrenamiento y evaluación de CNNs
    """
    def build_cnn(self, genome, dataset_params):
        """
        Construye arquitectura PyTorch desde genoma
        
        Pipeline:
        1. Construir capas convolucionales
        2. Construir capas fully connected
        3. Ensamblar en nn.Sequential
        """
        layers = []
        
        # Bloques convolucionales
        for i in range(genome['num_conv_layers']):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size),
                nn.BatchNorm2d(out_ch),
                activation_function(),
                nn.MaxPool2d(2, 2)
            ])
        
        # Capas fully connected
        layers.extend([
            nn.Flatten(),
            nn.Linear(flatten_size, fc_size),
            nn.Dropout(genome['dropout']),
            nn.Linear(fc_size, num_classes)
        ])
        
        return nn.Sequential(*layers)
    
    def train(self, model, train_loader, optimizer, criterion, epochs=3):
        """
        Entrena modelo durante N epochs
        """
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def evaluate(self, model, test_loader):
        """
        Evalúa accuracy en test set
        
        Returns:
            accuracy: float (0-100%)
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total
```

**Flujo de Trabajo**:
```
1. Recibir genoma desde Kafka
   └─ Topic: evolutioner-create-cnn-model

2. Cargar dataset
   └─ Si path especificado: ImageFolder(path)
   └─ Si no: MNIST (default)

3. Construir CNN
   └─ genome → PyTorch nn.Sequential

4. Entrenar (3 epochs)
   └─ Adam/SGD optimizer
   └─ CrossEntropyLoss
   └─ GPU si disponible

5. Evaluar
   └─ Accuracy en test set

6. Reportar fitness
   └─ Topic: evolutioner-create-cnn-model-response
   └─ {"model_id": "...", "score": 92.3, "uuid": "..."}
```

### 5. GeneticAlgorithmService

**Responsabilidades**:
```python
class GeneticAlgorithmService:
    """
    Motor del algoritmo genético multi-generacional
    """
    def run_genetic_algorithm(self, config):
        """
        Bucle evolutivo principal
        
        Pseudocódigo:
        
        población = create_initial_population(size=N)
        
        for generación in range(max_generations):
            # Evaluación
            fitness_scores = evaluate_population(población)
            
            # Verificar convergencia
            if fitness >= threshold:
                return best_model
            if stagnation_detected():
                return best_model
            
            # Selección (elitismo)
            élite = select_best(población, top_k=50%)
            
            # Reproducción
            hijos = []
            for i in range(N//2):
                padre1, padre2 = tournament_selection(élite)
                hijo = crossover(padre1, padre2)
                hijo = mutate(hijo, rate=mutation_rate)
                hijos.append(hijo)
            
            # Nueva generación
            población = élite + hijos
        
        return best_overall_model
        """
    
    def check_convergence(self, generation, fitness, history):
        """
        Criterios de parada
        
        Returns:
            (converged: bool, reason: str)
        """
        # 1. Máximo de generaciones
        if generation >= max_generations:
            return True, "Max generations"
        
        # 2. Threshold alcanzado
        if fitness >= fitness_threshold:
            return True, f"Threshold {fitness_threshold}% achieved"
        
        # 3. Estancamiento (no mejora en últimas 5 gen)
        if len(history) >= 5:
            recent = history[-5:]
            if max(recent) - min(recent) < 0.01:
                return True, "Stagnation detected"
        
        return False, "Continue"
```

### 6. PostgreSQL Database

**Esquema de Base de Datos**:

```sql
-- Tabla de poblaciones
CREATE TABLE populations (
    uuid VARCHAR(255) PRIMARY KEY,
    generation INTEGER NOT NULL,
    population_size INTEGER,
    population_data JSONB NOT NULL,    -- Array de genomas
    fitness_scores JSONB,               -- Array de accuracy scores
    best_fitness FLOAT,
    best_model_id VARCHAR(255),
    config JSONB,                       -- Configuración original
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_generation ON populations(generation);
CREATE INDEX idx_best_fitness ON populations(best_fitness DESC);

-- Tabla de evaluaciones individuales
CREATE TABLE model_evaluations (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR(255) REFERENCES populations(uuid),
    model_id VARCHAR(255) NOT NULL,
    architecture JSONB NOT NULL,        -- Genoma completo
    fitness FLOAT NOT NULL,             -- Accuracy
    training_time FLOAT,                -- Segundos
    num_parameters INTEGER,             -- Tamaño del modelo
    evaluated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fitness ON model_evaluations(fitness DESC);

-- Tabla de metadatos de ejecución
CREATE TABLE evolution_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_generations INTEGER,
    convergence_reason VARCHAR(255),
    best_overall_fitness FLOAT,
    best_overall_uuid VARCHAR(255),
    config JSONB
);
```

**Queries Importantes**:

```sql
-- Obtener mejor modelo histórico
SELECT model_id, fitness, architecture 
FROM model_evaluations 
ORDER BY fitness DESC 
LIMIT 1;

-- Tracking de evolución
SELECT generation, best_fitness 
FROM populations 
WHERE uuid LIKE 'run-123%' 
ORDER BY generation;

-- Análisis de convergencia
SELECT 
    generation,
    best_fitness,
    best_fitness - LAG(best_fitness) OVER (ORDER BY generation) AS improvement
FROM populations
WHERE uuid LIKE 'run-123%';

-- Distribución de arquitecturas
SELECT 
    architecture->>'num_conv_layers' AS conv_layers,
    AVG(fitness) AS avg_fitness,
    COUNT(*) AS count
FROM model_evaluations
GROUP BY conv_layers
ORDER BY avg_fitness DESC;
```

---

## 🔄 Flujos de Datos

### Flujo 1: Inicialización del Algoritmo Genético

```
Usuario
  │
  │ POST /start-genetic-algorithm
  │ {num_poblation: 20, max_generations: 50, ...}
  ▼
Broker
  │
  │ Kafka: genetic-algorithm
  ▼
GeneticAlgorithmService
  │
  │ Kafka: create-initial-population
  ▼
Genome Service
  │
  │ Genera 20 genomas aleatorios
  │
  │ Kafka: create-initial-population-response
  ▼
Broker
  │
  │ Almacena en PostgreSQL
  │ UUID: gen0-pop1
  ▼
PostgreSQL
```

### Flujo 2: Evaluación de Población

```
GeneticAlgorithmService
  │
  │ Kafka: evaluate-population
  │ {uuid: "gen0-pop1"}
  ▼
Broker
  │
  │ Lee población desde PostgreSQL
  │ Obtiene 20 genomas
  ▼
PostgreSQL
  │
  │ Para cada genoma (i=1..20):
  │   Kafka: evolutioner-create-cnn-model
  │   {genome: {...}, model_id: "model-i"}
  ▼
Evolutioners (múltiples instancias en paralelo)
  │
  │ Instancia 1: Modelos 1-5
  │ Instancia 2: Modelos 6-10
  │ Instancia 3: Modelos 11-15
  │ Instancia 4: Modelos 16-20
  │
  │ Cada instancia:
  │   1. Construir CNN
  │   2. Cargar dataset
  │   3. Entrenar 3 epochs
  │   4. Evaluar accuracy
  │
  │ Kafka: evolutioner-create-cnn-model-response
  │ {model_id: "model-i", score: 85.3}
  ▼
Broker
  │
  │ Recolecta todos los scores
  │ fitness_scores = [78.2, 85.3, 67.1, ...]
  │
  │ Actualiza en PostgreSQL
  ▼
PostgreSQL
  │
  │ Kafka: evaluate-population-response
  │ {uuid: "gen0-pop1", fitness_scores: [...]}
  ▼
GeneticAlgorithmService
```

### Flujo 3: Selección y Reproducción

```
GeneticAlgorithmService
  │
  │ fitness_scores = [78.2, 85.3, 92.1, 67.1, ...]
  │
  │ Kafka: select-best-architectures
  │ {uuid: "gen0-pop1", fitness_scores: [...]}
  ▼
Broker
  │
  │ Ordena por fitness (descendente)
  │ Selecciona top 50% (élite)
  │ elite = [model-3: 92.1, model-2: 85.3, ...]
  │
  │ Kafka: select-best-architectures-response
  ▼
GeneticAlgorithmService
  │
  │ Para crear 10 nuevos hijos:
  │   Tournament selection → padre1, padre2
  │
  │   Kafka: genome-create-child
  │   {parent1: model-3, parent2: model-2}
  ▼
Genome Service
  │
  │ 1. Crossover(padre1, padre2)
  │    - Combinar genomas
  │ 2. Mutate(hijo, rate=0.1)
  │    - Mutaciones aleatorias
  │
  │ Kafka: genome-create-child-response
  │ {child_genome: {...}}
  ▼
Broker
  │
  │ Nueva población = élite + hijos
  │ población_gen1 = [model-3, model-2, ..., hijo-1, hijo-2, ...]
  │
  │ Almacena en PostgreSQL
  │ UUID: gen1-pop1
  ▼
PostgreSQL
  │
  │ Kafka: continue-algorithm
  │ {uuid: "gen1-pop1", generation: 1}
  ▼
GeneticAlgorithmService
  │
  │ Verificar convergencia:
  │   - generation < max_generations? ✓
  │   - best_fitness < threshold? ✓
  │   - stagnation? ✗
  │
  │ → CONTINUAR con Generación 2
  │ → LOOP a "Evaluación de Población"
```

### Flujo 4: Convergencia y Finalización

```
GeneticAlgorithmService
  │
  │ Generación 23:
  │   best_fitness = 95.3%
  │   threshold = 95.0%
  │
  │ check_convergence() → (True, "Threshold achieved")
  ▼
Broker
  │
  │ Obtener mejor modelo de todas las generaciones
  │ SELECT * FROM model_evaluations ORDER BY fitness DESC LIMIT 1
  ▼
PostgreSQL
  │
  │ best_model = {
  │   uuid: "gen23-pop1",
  │   model_id: "model-5",
  │   fitness: 95.3,
  │   architecture: {...}
  │ }
  │
  │ Kafka: genetic-algorithm-response
  │ {
  │   status: "success",
  │   converged: true,
  │   reason: "Threshold 95.0% achieved",
  │   best_model: {...},
  │   generations: 23,
  │   fitness_history: [65.3, 72.1, ..., 95.3]
  │ }
  ▼
Usuario
```

---

## 📊 Diagramas de Arquitectura

### Diagrama de Despliegue

```
┌─────────────────────────────────────────────────────────────────┐
│                          DOCKER HOST                             │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Zookeeper   │  │    Kafka     │  │  PostgreSQL  │          │
│  │   :2181      │◄─┤   :9092      │  │    :5432     │          │
│  └──────────────┘  └──────┬───────┘  └──────────────┘          │
│                            │                                     │
│  ┌─────────────────────────┼────────────────────────┐           │
│  │                         │                        │           │
│  ▼                         ▼                        ▼           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Broker     │  │    Genome    │  │ Evolutioners │          │
│  │  Container   │  │  Container   │  │  Container   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │         GeneticAlgorithmService Container        │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Diagrama de Clases Simplificado

```python
class BrokerService:
    - consumer: KafkaConsumer
    - producer: KafkaProducer
    - db: PostgreSQLConnection
    
    + handle_create_population()
    + handle_evaluate_population()
    + handle_select_best()
    + save_to_database()

class GenomeService:
    - consumer: KafkaConsumer
    - producer: KafkaProducer
    
    + create_initial_population(n: int) -> List[Genome]
    + crossover(parent1: Genome, parent2: Genome) -> Genome
    + mutate(genome: Genome, rate: float) -> Genome

class EvolutionersService:
    - consumer: KafkaConsumer
    - producer: KafkaProducer
    - device: torch.device
    
    + build_cnn(genome: Genome, params: Dict) -> nn.Module
    + train(model: nn.Module, loader: DataLoader) -> None
    + evaluate(model: nn.Module, loader: DataLoader) -> float

class GeneticAlgorithmService:
    - consumer: KafkaConsumer
    - producer: KafkaProducer
    
    + run_algorithm(config: Dict) -> Dict
    + check_convergence(...) -> Tuple[bool, str]
```

---

## 🎯 Decisiones de Diseño

### ¿Por qué Kafka?

**Alternativas consideradas**: RabbitMQ, Redis Pub/Sub, AWS SQS

**Razones para elegir Kafka**:
1. ✅ **Throughput**: Millones de mensajes/segundo
2. ✅ **Persistencia**: Mensajes se almacenan en disco
3. ✅ **Replay**: Posibilidad de re-procesar mensajes históricos
4. ✅ **Escalabilidad**: Particionamiento para paralelización
5. ✅ **Ecosistema**: Integración con muchas herramientas

**Trade-offs**:
- ❌ Más complejo de configurar que RabbitMQ
- ❌ Mayor overhead de recursos (Zookeeper requerido)
- ✅ Pero mejor para nuestro caso de uso (alto volumen)

### ¿Por qué PostgreSQL?

**Alternativas consideradas**: MongoDB, DynamoDB, MySQL

**Razones**:
1. ✅ **JSONB**: Almacenar genomas como JSON flexible
2. ✅ **Transacciones ACID**: Consistencia garantizada
3. ✅ **Índices en JSON**: Queries rápidas en estructuras complejas
4. ✅ **Maduro y robusto**: Battle-tested
5. ✅ **Open source**: Sin costos de licencia

### ¿Por qué Microservicios?

**Alternativa**: Monolito con threads/procesos

**Ventajas de microservicios**:
1. ✅ **Escalabilidad independiente**: Más Evolutioners sin afectar otros
2. ✅ **Despliegue independiente**: Actualizar Genome sin reiniciar todo
3. ✅ **Tolerancia a fallos**: Fallo aislado no colapsa sistema
4. ✅ **Tecnologías heterogéneas**: Posibilidad de usar otros lenguajes
5. ✅ **Desarrollo paralelo**: Equipos trabajan independientemente

**Trade-offs**:
- ❌ Mayor complejidad operacional
- ❌ Latencia de comunicación entre servicios
- ❌ Debugging más difícil
- ✅ Pero crítico para escalabilidad que necesitamos

### ¿Por qué PyTorch?

**Alternativas**: TensorFlow, JAX

**Razones**:
1. ✅ **Pythonic**: API más intuitiva
2. ✅ **Dynamic graphs**: Facilita debugging
3. ✅ **Comunidad research**: Cutting-edge algorithms
4. ✅ **TorchVision**: Utilidades para imágenes
5. ✅ **CUDA support**: Excelente integración GPU

---

## 🔒 Consideraciones de Seguridad

### Datos Sensibles

```python
# ❌ NUNCA hardcodear credenciales
POSTGRES_PASSWORD = "neat_pass"

# ✅ Usar variables de entorno
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

# ✅ O archivos de secrets
with open("/run/secrets/postgres_password") as f:
    POSTGRES_PASSWORD = f.read().strip()
```

### Validación de Inputs

```python
def validate_config(config):
    """Validar configuración de usuario"""
    assert 1 <= config['num_poblation'] <= 1000, "Invalid population size"
    assert 1 <= config['max_generations'] <= 500, "Invalid generations"
    assert 0 <= config['mutation_rate'] <= 1.0, "Invalid mutation rate"
    assert config['num_classes'] >= 2, "Need at least 2 classes"
    
    # Sanitizar path
    if 'path' in config:
        path = Path(config['path']).resolve()
        assert path.exists(), f"Path {path} does not exist"
        assert path.is_dir(), f"Path {path} is not a directory"
```

### Rate Limiting

```python
# Limitar número de requests simultáneos
MAX_CONCURRENT_EVOLUTIONS = 5

if len(active_evolutions) >= MAX_CONCURRENT_EVOLUTIONS:
    return {"error": "Too many concurrent evolutions"}
```

---

## 📈 Monitoreo y Observabilidad

### Métricas Clave

```python
# Prometheus metrics (ejemplo)
evolution_duration = Histogram('evolution_duration_seconds')
models_trained_total = Counter('models_trained_total')
current_best_fitness = Gauge('current_best_fitness')
kafka_lag = Gauge('kafka_consumer_lag')
```

### Logging Estructurado

```python
import logging
import json

logger.info(json.dumps({
    'event': 'model_trained',
    'model_id': 'gen5-model3',
    'fitness': 92.3,
    'training_time': 45.2,
    'generation': 5,
    'timestamp': datetime.now().isoformat()
}))
```

### Health Checks

```python
@app.route('/health')
def health_check():
    checks = {
        'kafka': check_kafka_connection(),
        'postgres': check_db_connection(),
        'gpu': torch.cuda.is_available()
    }
    
    status = 'healthy' if all(checks.values()) else 'unhealthy'
    return jsonify({'status': status, 'checks': checks})
```

---

## 🚀 Optimizaciones Futuras

### 1. Caché de Modelos

```python
# Redis cache para evitar re-entrenar modelos idénticos
cache_key = hash(json.dumps(genome, sort_keys=True))
if cache_key in redis_cache:
    return redis_cache[cache_key]
```

### 2. Distributed Training

```python
# PyTorch DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### 3. Early Stopping Inteligente

```python
# Detener entrenamiento si accuracy no mejora
if best_val_acc not in top_k_models:
    return early_stop_fitness  # No malgastar recursos
```

### 4. Adaptive Mutation Rate

```python
# Aumentar mutación si hay estancamiento
if stagnation_detected():
    mutation_rate *= 1.5
else:
    mutation_rate *= 0.95
```

---

**Autor**: Proyecto Neuroevolution  
**Fecha**: Octubre 2025  
**Versión**: 1.0
