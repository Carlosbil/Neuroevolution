# 🚀 Guía de Inicio Rápido - Neuroevolution

## ⏱️ 5 Minutos para Empezar

### 1️⃣ Iniciar el Sistema (Docker)

```bash
# Clonar repositorio
git clone https://github.com/Carlosbil/Neuroevolution.git
cd Neuroevolution

# Iniciar todos los servicios
docker-compose up -d

# Verificar que están corriendo
docker-compose ps
```

**Servicios que deberías ver**:
- ✅ zookeeper (puerto 2181)
- ✅ kafka (puerto 9092)
- ✅ postgres (puerto 5432)
- ✅ genome

### 2️⃣ Probar con MNIST (Sin preparación de datos)

```bash
# Crear archivo test_mnist.py
cat > test_mnist.py << 'EOF'
import json
from confluent_kafka import Producer

producer = Producer({'bootstrap.servers': 'localhost:9092'})

config = {
    'num_channels': 1,
    'px_h': 28,
    'px_w': 28,
    'num_classes': 10,
    'batch_size': 64,
    'num_poblation': 5,      # Solo 5 individuos para test rápido
    'max_generations': 3,     # Solo 3 generaciones
    'fitness_threshold': 85.0
}

producer.produce('genetic-algorithm', json.dumps(config))
producer.flush()
print("✅ Algoritmo genético iniciado!")
print("📊 Ver logs: docker-compose logs -f")
EOF

# Ejecutar
python test_mnist.py
```

**Tiempo esperado**: ~15-20 minutos

**Resultado esperado**:
```
Generación 0: ~60% accuracy (arquitecturas aleatorias)
Generación 1: ~75% accuracy
Generación 2: ~85% accuracy ✅ (threshold alcanzado)
```

### 3️⃣ Ver Progreso

```bash
# Ver logs en tiempo real
docker-compose logs -f genetic-algorithm
docker-compose logs -f evolutioners

# Ver mejor modelo en base de datos
docker exec -it postgres psql -U neat_user -d neat_db -c \
  "SELECT generation, best_fitness FROM populations ORDER BY best_fitness DESC LIMIT 5;"
```

---

## 📊 Flujo Visual del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│  USUARIO: Envía configuración                               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  KAFKA: Recibe mensaje en topic 'genetic-algorithm'         │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  GENETIC ALGORITHM SERVICE                                   │
│  - Crea población inicial (N arquitecturas aleatorias)      │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  GENOME SERVICE                                              │
│  - Genera genomas aleatorios                                 │
│  - Ejemplo: {conv_layers: 3, filters: [32,64,128], ...}    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  EVOLUTIONERS (ENTRENAMIENTO)                                │
│  Para cada arquitectura en paralelo:                         │
│    1. Construir CNN desde genoma                             │
│    2. Entrenar 3 epochs                                      │
│    3. Evaluar accuracy                                       │
│  Resultado: [78%, 65%, 82%, 91%, 73%]                       │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  SELECCIÓN + REPRODUCCIÓN                                    │
│  1. Seleccionar mejores (top 50%): [91%, 82%, 78%]         │
│  2. Crossover: Combinar padres                               │
│  3. Mutación: Cambios aleatorios                             │
│  4. Nueva generación → LOOP                                  │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  CONVERGENCIA                                                │
│  ¿Alcanzó threshold? ✅ → FIN                               │
│  ¿Máx generaciones? ✅ → FIN                                │
│  ¿Estancamiento? ✅ → FIN                                    │
│  Si no → Continuar con siguiente generación                 │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESULTADO FINAL                                             │
│  - UUID del mejor modelo                                     │
│  - Accuracy: 95.3%                                           │
│  - Arquitectura evolucionada                                 │
│  - Historia de fitness por generación                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Casos de Uso Rápidos

### Caso A: Clasificación de Imágenes Genérica

```python
config = {
    'num_channels': 3,              # RGB
    'px_h': 224,                    # Tamaño imagen
    'px_w': 224,
    'num_classes': 5,               # Número de clases
    'batch_size': 32,
    'num_poblation': 15,
    'max_generations': 30,
    'fitness_threshold': 90.0,
    'path': '/ruta/a/mi/dataset'    # Estructura: dataset/clase_A/img1.jpg
}
```

### Caso B: Detección de Parkinson (Audio)

```bash
# 1. Convertir audio a espectrogramas
python wav_to_images_converter.py
# Ruta: ./data/parkinson_audio
# GPU: Sí
# Tipo: Solo espectrogramas

# 2. Configurar y ejecutar
config = {
    'num_channels': 1,              # Espectrogramas grayscale
    'px_h': 128,
    'px_w': 128,
    'num_classes': 2,               # Sano vs Parkinson
    'batch_size': 32,
    'num_poblation': 20,
    'max_generations': 50,
    'fitness_threshold': 95.0,
    'path': './data/parkinson_audio/images_combined'
}
```

### Caso C: Prueba Rápida (1 minuto)

```python
# Solo verificar que todo funciona
config = {
    'num_channels': 1,
    'px_h': 28,
    'px_w': 28,
    'num_classes': 10,
    'batch_size': 128,
    'num_poblation': 2,        # Mínimo
    'max_generations': 1,      # 1 generación
    'fitness_threshold': 50.0  # Bajo
    # Sin 'path' → usa MNIST
}
```

---

## 🔍 Cheatsheet de Comandos

### Docker

```bash
# Iniciar todo
docker-compose up -d

# Detener todo
docker-compose down

# Ver logs
docker-compose logs -f [servicio]

# Reiniciar servicio
docker-compose restart genome

# Ver recursos
docker stats

# Limpiar todo (¡cuidado!)
docker-compose down -v  # -v elimina volúmenes (base de datos)
```

### Kafka

```bash
# Listar topics
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# Ver mensajes de un topic
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic genetic-algorithm \
  --from-beginning

# Crear topic manualmente
docker exec -it kafka kafka-topics \
  --create \
  --bootstrap-server localhost:9092 \
  --topic mi-nuevo-topic \
  --partitions 1 \
  --replication-factor 1
```

### PostgreSQL

```bash
# Conectar a base de datos
docker exec -it postgres psql -U neat_user -d neat_db

# Consultas útiles
SELECT * FROM populations ORDER BY created_at DESC LIMIT 5;
SELECT generation, best_fitness FROM populations WHERE uuid LIKE 'run%';
SELECT COUNT(*) FROM model_evaluations;
```

### Python

```bash
# Ver todos los servicios de Python
ps aux | grep python

# Matar proceso
kill -9 [PID]

# Ver uso de GPU
nvidia-smi

# Ver uso de memoria
free -h
```

---

## ⚠️ Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| ❌ Kafka no conecta | `docker-compose restart kafka zookeeper` |
| ❌ "Out of memory" | Reducir `batch_size` o `num_poblation` |
| ❌ GPU no detectada | Verificar `nvidia-docker` instalado |
| ❌ Dataset no encontrado | Verificar que `path` sea ruta absoluta |
| ❌ Proceso muy lento | Verificar que GPU está siendo usada: `torch.cuda.is_available()` |
| ❌ Topic no existe | Revisar `docker-compose logs kafka-init` |

### Verificación de Salud

```python
# health_check.py
import torch
from confluent_kafka import Producer
import psycopg2

# 1. PyTorch + CUDA
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# 2. Kafka
try:
    p = Producer({'bootstrap.servers': 'localhost:9092'})
    print("✅ Kafka OK")
except:
    print("❌ Kafka Error")

# 3. PostgreSQL
try:
    conn = psycopg2.connect(
        host='localhost', 
        database='neat_db',
        user='neat_user', 
        password='neat_pass'
    )
    print("✅ PostgreSQL OK")
    conn.close()
except:
    print("❌ PostgreSQL Error")
```

---

## 📚 Siguientes Pasos

1. **Leer documentación detallada**:
   - [`README.md`](./README.md) - Documentación completa
   - [`CNN_TIME_SERIES.md`](./CNN_TIME_SERIES.md) - Cómo funcionan las CNNs para series temporales
   - [`ARCHITECTURE.md`](./ARCHITECTURE.md) - Arquitectura del sistema

2. **Experimentar con parámetros**:
   - Probar diferentes `num_poblation` (5, 10, 20, 50)
   - Ajustar `mutation_rate` (0.05, 0.1, 0.2)
   - Modificar `fitness_threshold`

3. **Usar tu propio dataset**:
   - Organizar imágenes en carpetas por clase
   - Ejecutar con `path` apuntando a tu dataset

4. **Analizar resultados**:
   - Consultar base de datos PostgreSQL
   - Ver evolución de fitness
   - Comparar arquitecturas

---

## 💡 Tips Pro

### Acelerar Entrenamiento

```python
config = {
    ...
    'batch_size': 128,        # ↑ Más grande si tienes GPU potente
    'num_poblation': 50,      # ↑ Más exploración, pero más lento
}
```

### Mejor Exploración

```python
config = {
    ...
    'mutation_rate': 0.2,     # ↑ Más variedad
    'max_generations': 100,   # ↑ Más tiempo para converger
}
```

### Modo "Quick and Dirty"

```python
# Para desarrollo rápido
config = {
    'num_channels': 1,
    'px_h': 28,
    'px_w': 28,
    'num_classes': 10,
    'batch_size': 256,        # Grande para velocidad
    'num_poblation': 3,       # Mínimo
    'max_generations': 5,     # Pocas generaciones
    'fitness_threshold': 80.0 # Bajo
}
# Tiempo: ~5 minutos en GPU
```

---

## 🎓 Conceptos Clave en 30 Segundos

**Neuroevolución**: Usar algoritmos genéticos para descubrir arquitecturas de redes neuronales óptimas automáticamente.

**Genoma**: Especificación de una arquitectura CNN (número de capas, filtros, activaciones, etc.)

**Fitness**: Qué tan bueno es un modelo (accuracy en nuestro caso)

**Generación**: Conjunto de N arquitecturas evaluadas simultáneamente

**Crossover**: Combinar dos arquitecturas "padre" para crear un "hijo"

**Mutación**: Cambios aleatorios en una arquitectura

**Convergencia**: Cuando el algoritmo encuentra una solución suficientemente buena y se detiene

---

**¿Listo para empezar?** Ejecuta:

```bash
docker-compose up -d && python test_mnist.py
```

**¿Tienes dudas?** Consulta [`README.md`](./README.md) o abre un [Issue en GitHub](https://github.com/Carlosbil/Neuroevolution/issues)

---

🌟 **Happy Evolving!** 🌟
