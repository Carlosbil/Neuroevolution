# Referencia para Paper de Investigación: Neuroevolución Híbrida para Clasificación de Audio en Detección de Parkinson

## Información del Documento

| Campo | Valor |
|-------|-------|
| **Archivo Fuente** | `best_Audio_hybrid_neuroevolution_notebook.ipynb` |
| **Fecha de Generación** | Diciembre 2024 |
| **Dominio** | Machine Learning, Neuroevolución, Clasificación de Audio Médico |
| **Aplicación** | Detección de Parkinson mediante análisis de voz |

---

## 1. Resumen Ejecutivo (Abstract)

### 1.1 Objetivo
Desarrollo de un sistema de **neuroevolución híbrida** para la optimización automática de arquitecturas de redes neuronales convolucionales 1D (Conv1D) aplicadas a la **clasificación de audio para detección de Parkinson**.

### 1.2 Contribuciones Principales
1. **Algoritmo genético híbrido** que combina evolución de arquitectura y pesos
2. **Validación cruzada paralela de 5-fold** durante el proceso evolutivo
3. **Tasa de mutación adaptativa** basada en la diversidad poblacional
4. **Soporte para datos reales y sintéticos** (generados por GANs)
5. **Sistema de checkpoint** para preservar el mejor modelo global

### 1.3 Resultados Clave
- Fitness objetivo: **≥80%** de accuracy
- Arquitectura: Redes **Conv1D** optimizadas para señales de audio 1D
- Clases: **Control vs Patológico** (clasificación binaria)

---

## 2. Metodología

### 2.1 Algoritmo Genético Híbrido

#### 2.1.1 Representación del Genoma
Cada individuo (genoma) representa una arquitectura de red neuronal con los siguientes genes:

```python
genome = {
    # Estructura de la red
    'num_conv_layers': int,      # Número de capas convolucionales (1-30)
    'num_fc_layers': int,        # Número de capas fully connected (1-10)
    'filters': List[int],        # Filtros por capa conv (1-256)
    'kernel_sizes': List[int],   # Tamaños de kernel (1,3,5,7,9,11,13,15)
    'fc_nodes': List[int],       # Neuronas por capa FC (64-1024)
    
    # Hiperparámetros
    'activations': List[str],    # Funciones de activación
    'dropout_rate': float,       # Tasa de dropout (0.2-0.6)
    'learning_rate': float,      # Learning rate
    'optimizer': str,            # Optimizador (adam, adamw, sgd, rmsprop)
    'normalization_type': str,   # batch o layer normalization
    
    # Metadatos
    'fitness': float,            # Accuracy promedio 5-fold
    'id': str                    # Identificador único
}
```

#### 2.1.2 Parámetros del Algoritmo Genético

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `population_size` | 20 | Tamaño de la población |
| `max_generations` | 100 | Máximo de generaciones |
| `fitness_threshold` | 80.0% | Objetivo de fitness |
| `base_mutation_rate` | 0.25 | Tasa de mutación inicial |
| `mutation_rate_min` | 0.10 | Límite inferior adaptativo |
| `mutation_rate_max` | 0.80 | Límite superior adaptativo |
| `crossover_rate` | 0.99 | Tasa de crossover |
| `elite_percentage` | 0.20 | Porcentaje de élite (20%) |

### 2.2 Operadores Genéticos

#### 2.2.1 Creación de Genoma Aleatorio
- Calcula el **máximo seguro de capas conv** basado en la longitud de secuencia
- Validación para evitar arquitecturas inválidas (dimensiones demasiado reducidas)
- Incremento progresivo de filtros en capas más profundas

#### 2.2.2 Mutación Adaptativa
```python
# Fórmula de mutación adaptativa
diversity_factor = min(1.0, std_fitness / 10.0)
inverted = 1 - diversity_factor
new_rate = base_rate + (inverted - 0.5) * 0.4
```

**Mutaciones posibles:**
- Número de capas (conv y FC)
- Filtros por capa
- Tamaños de kernel
- Neuronas en capas FC
- Funciones de activación
- Dropout rate
- Learning rate
- Optimizador
- Tipo de normalización

#### 2.2.3 Crossover
- Crossover de parámetros escalares con probabilidad 50%
- Crossover de listas con punto de corte aleatorio
- Validación post-crossover para garantizar arquitecturas válidas

#### 2.2.4 Selección
- **Selección por torneo** (tournament_size = 3)
- **Elitismo moderado** (20% de la población)

### 2.3 Validación Cruzada Paralela de 5-Fold

#### 2.3.1 Proceso de Evaluación
```
Para cada individuo:
  1. ThreadPool con 5 workers
  2. Fold 1-5: Entrenamiento SIMULTÁNEO
  3. Espera: Todos los threads completan
  4. Fitness = promedio(accuracy_1, ..., accuracy_5)
```

#### 2.3.2 Ventajas del Enfoque Paralelo
| Ventaja | Descripción |
|---------|-------------|
| Velocidad | ~5x más rápido que evaluación secuencial |
| Robustez | Evita sobreajuste a un fold específico |
| Generalización | La arquitectura seleccionada generaliza mejor |
| Eficiencia | Aprovecha múltiples núcleos de CPU |

### 2.4 Criterios de Convergencia

1. **Target fitness alcanzado**: fitness ≥ 80%
2. **Máximo de generaciones**: generation ≥ 100
3. **Early stopping**: Sin mejora significativa (≥0.01%) en 20 generaciones
4. **Estancamiento**: Variación < 0.5% en últimas 3 generaciones

---

## 3. Arquitectura de Red Neuronal

### 3.1 Estructura General
```
Input (1D Audio Signal)
    ↓
[Conv1D → BatchNorm1D → Activation → MaxPool1D] × N
    ↓
Flatten
    ↓
[FC → BatchNorm1D → ReLU → Dropout] × M
    ↓
Output Layer (2 clases)
```

### 3.2 Detalles de Capas Convolucionales (Conv1D)

| Componente | Descripción |
|------------|-------------|
| Conv1D | Kernel sizes: 1, 3, 5, 7, 9, 11, 13, 15 |
| Normalización | BatchNorm1D (80%) o LayerNorm (20%) |
| Activación | ReLU, LeakyReLU, Tanh, Sigmoid, SELU |
| Pooling | MaxPool1D(2, 2) |
| Dropout | 0.1 entre capas conv |

### 3.3 Detalles de Capas Fully Connected

| Componente | Descripción |
|------------|-------------|
| Linear | 64-1024 neuronas |
| Normalización | BatchNorm1D o LayerNorm |
| Activación | ReLU |
| Dropout | 0.2-0.6 (configurable) |

### 3.4 Validación de Arquitectura
- **Problema**: BatchNorm1d requiere más de 1 valor en dimensión espacial
- **Solución**: Validación previa del número máximo seguro de capas
```python
max_safe_conv_layers = log2(sequence_length / min_required_length)
```

---

## 4. Configuración del Dataset

### 4.1 Datos de Audio para Parkinson

| Parámetro | Valor |
|-----------|-------|
| Tipo | Forma de onda 1D |
| Canales | 1 |
| Longitud de secuencia | Auto-detectada (~240,000 muestras) |
| Clases | 2 (Control vs Patológico) |
| Formato | Archivos `.npy` |

### 4.2 Estructura de Datos (5-Fold)
```
data/sets/folds_5/files_real_{fold_id}/
├── X_train_{dataset_id}_fold_{1-5}.npy
├── y_train_{dataset_id}_fold_{1-5}.npy
├── X_val_{dataset_id}_fold_{1-5}.npy
├── y_val_{dataset_id}_fold_{1-5}.npy
├── X_test_{dataset_id}_fold_{1-5}.npy
└── y_test_{dataset_id}_fold_{1-5}.npy
```

### 4.3 Tipos de Dataset Disponibles

| Carpeta | Train | Test | Uso |
|---------|-------|------|-----|
| `files_real_N` | Reales | Reales | Baseline |
| `files_real_40_1e5_N` | Sintéticos (GAN) | Reales | Transfer learning |
| `files_syn_40_1e5_N` | Sintéticos | Sintéticos | Capacidad sintética |
| `files_syn_1_N` | Sintéticos | Sintéticos (diferentes) | Generalización |
| `files_all_real_syn_n` | **Mixto (Real+Sintético)** | **Mixto** | **Mejor generalización** |

### 4.4 Datos Sintéticos (GANs)
- Generados con **BigVSAN** (configuración 40_1e5)
- Aumentan diversidad de entrenamiento
- Combinan autenticidad (reales) con variedad (sintéticos)

---

## 5. Parámetros de Entrenamiento

### 5.1 Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Batch size | 64 |
| Max epochs | 100 |
| Learning rate base | 0.00001 |
| Early stopping (épocas) | Patience = 10 |
| Improvement threshold | 0.01% |

### 5.2 Optimizadores Disponibles

| Optimizador | Clase PyTorch |
|-------------|---------------|
| Adam | `torch.optim.Adam` |
| AdamW | `torch.optim.AdamW` |
| SGD | `torch.optim.SGD` |
| RMSprop | `torch.optim.RMSprop` |

### 5.3 Learning Rates Disponibles
`[0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.01, 0.1]`

### 5.4 Funciones de Activación

| Nombre | Clase PyTorch |
|--------|---------------|
| relu | `nn.ReLU` |
| leaky_relu | `nn.LeakyReLU` |
| tanh | `nn.Tanh` |
| sigmoid | `nn.Sigmoid` |
| selu | `nn.SELU` |

---

## 6. Sistema de Checkpoints

### 6.1 Gestión de Checkpoints
- **Guardado automático** cuando se encuentra un nuevo mejor global
- **Eliminación automática** del checkpoint anterior
- **Ubicación**: `checkpoints/best_model_gen{X}_id{Y}_fitness{Z}.pth`

### 6.2 Contenido del Checkpoint
```python
checkpoint_data = {
    'model_state_dict': model.state_dict(),
    'genome': genome,
    'generation': generation,
    'fitness': fitness,
    'config': config
}
```

---

## 7. Métricas y Visualización

### 7.1 Métricas Durante Evolución
- **Fitness máximo** por generación
- **Fitness promedio** por generación
- **Fitness mínimo** por generación
- **Desviación estándar** (diversidad)

### 7.2 Métricas Finales (5-Fold CV)
- Accuracy
- Sensitivity (Recall)
- Specificity
- F1-Score
- AUC (Area Under Curve)

### 7.3 Visualizaciones Generadas
1. **Evolución del Fitness**: Curvas de max/avg/min fitness
2. **Diversidad Poblacional**: Desviación estándar por generación
3. **Análisis de Fallos**: Evaluaciones con fitness 0.00

---

## 8. Implementación Técnica

### 8.1 Dependencias Principales

```python
# Core
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.21.0

# Visualización
matplotlib >= 3.5.0
seaborn >= 0.11.0

# Utilidades
tqdm >= 4.64.0
jupyter >= 1.0.0
```

### 8.2 Configuración de Hardware
- **GPU**: CUDA si está disponible
- **CPU**: Multi-threading para 5-fold paralelo
- **Memoria**: Pin memory para transferencia GPU

### 8.3 Reproducibilidad
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

---

## 9. Flujo de Ejecución

### 9.1 Pipeline Principal

```
1. Instalación de dependencias
2. Configuración de parámetros (CONFIG)
3. Carga y verificación del dataset
4. Inicialización de población
5. Loop evolutivo:
   a. Evaluación paralela 5-fold
   b. Verificación de convergencia
   c. Actualización de mutación adaptativa
   d. Selección y reproducción
6. Visualización de resultados
7. Guardado de mejor arquitectura (JSON)
```

### 9.2 Pseudocódigo del Algoritmo

```python
def evolve():
    initialize_population()
    
    while not converged():
        for individual in population:
            fitness = parallel_5fold_evaluation(individual)
            
            if fitness > global_best_fitness:
                save_checkpoint(individual)
        
        update_adaptive_mutation_rate()
        
        elite = select_elite(population)
        offspring = crossover_and_mutate(population)
        population = elite + offspring
        
        generation += 1
    
    return best_individual
```

---

## 10. Resultados Esperados

### 10.1 Formato de Salida JSON
```json
{
    "timestamp": "20241204_123456",
    "execution_time": "HH:MM:SS",
    "dataset_type": "audio_1D",
    "dataset_id": "40_1e5_N",
    "config_used": {...},
    "best_genome": {
        "id": "abc12345",
        "fitness": 85.67,
        "num_conv_layers": 5,
        "num_fc_layers": 2,
        ...
    },
    "final_generation": 50,
    "evolution_stats": [...]
}
```

### 10.2 Métricas de Rendimiento Objetivo

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| Accuracy | ≥80% | Precisión general |
| Sensitivity | Alto | Detección de casos positivos |
| Specificity | Alto | Detección de casos negativos |
| F1-Score | ≥0.75 | Balance precision/recall |

---

## 11. Consideraciones para el Paper

### 11.1 Contribuciones Científicas Destacables

1. **Neuroevolución con validación cruzada paralela**: Enfoque novedoso que combina eficiencia computacional con robustez estadística.

2. **Mutación adaptativa**: Ajuste dinámico de la tasa de mutación basado en diversidad poblacional.

3. **Soporte para datos sintéticos (GANs)**: Integración de datos generados por BigVSAN para augmentación.

4. **Validación de arquitectura**: Sistema de prevención de arquitecturas inválidas antes de evaluación.

5. **Sistema de checkpoints**: Preservación continua del mejor modelo para transfer learning.

### 11.2 Comparaciones Sugeridas

- Comparar con arquitecturas Conv1D estáticas (ResNet-1D, VGG-1D)
- Comparar con métodos tradicionales de features de audio (MFCCs + SVM/RF)
- Comparar tiempos de búsqueda con NAS clásico (sin paralelización)
- Evaluar impacto de datos sintéticos vs solo reales

### 11.3 Limitaciones a Mencionar

1. Tiempo computacional alto para poblaciones grandes
2. Dependencia de la calidad de datos sintéticos
3. Configuración de hiperparámetros del GA requiere tuning inicial
4. Validación en un solo dataset (Parkinson)

### 11.4 Trabajo Futuro Sugerido

1. Extensión a otros tipos de audio médico
2. Incorporación de transfer learning entre arquitecturas
3. Co-evolución de arquitectura y preprocesamiento
4. Aplicación a datos multimodales (audio + imágenes)

---

## 12. Referencias de Código Clave

### 12.1 Archivos Principales
- `best_Audio_hybrid_neuroevolution_notebook.ipynb`: Notebook principal
- `train_best_audio_model.py`: Script de entrenamiento del mejor modelo
- `generating_csv/create_5_folds.ipynb`: Generación de folds

### 12.2 Clases Principales
- `EvolvableCNN`: Red neuronal evolucionable (Conv1D)
- `HybridNeuroevolution`: Orquestador del algoritmo genético

### 12.3 Funciones Clave
- `create_random_genome()`: Creación de genomas aleatorios
- `mutate_genome()`: Operador de mutación
- `crossover_genomes()`: Operador de crossover
- `evaluate_fitness()`: Evaluación paralela 5-fold
- `is_genome_valid()`: Validación de arquitectura

---

## 13. Estructura del Paper Sugerida

1. **Abstract**
2. **Introduction**
   - Problema de clasificación de Parkinson por voz
   - Limitaciones de arquitecturas manuales
   - Propuesta: neuroevolución híbrida
3. **Related Work**
   - Neural Architecture Search (NAS)
   - Audio classification for Parkinson
   - Genetic algorithms in deep learning
4. **Methodology**
   - Representación del genoma
   - Operadores genéticos
   - Validación cruzada paralela
   - Mutación adaptativa
5. **Experimental Setup**
   - Dataset description
   - Configuration parameters
   - Evaluation metrics
6. **Results**
   - Evolution curves
   - Best architecture analysis
   - Comparison with baselines
7. **Discussion**
   - Impact of synthetic data
   - Efficiency of parallel evaluation
   - Generalization capabilities
8. **Conclusion**
9. **Future Work**
10. **References**

---

## 14. Glosario

| Término | Definición |
|---------|------------|
| **Genoma** | Representación codificada de una arquitectura de red neuronal |
| **Fitness** | Medida de calidad (accuracy promedio 5-fold) |
| **Elitismo** | Preservación de mejores individuos entre generaciones |
| **Crossover** | Combinación de genes de dos padres |
| **Mutación** | Modificación aleatoria de genes |
| **Conv1D** | Convolución 1D para datos secuenciales |
| **5-Fold CV** | Validación cruzada con 5 particiones |
| **BigVSAN** | GAN para generación de audio sintético |
| **BatchNorm** | Normalización por batch para estabilizar entrenamiento |

---

*Documento generado automáticamente como referencia para paper de investigación.*
*Para preguntas específicas sobre implementación, consultar el notebook fuente.*
