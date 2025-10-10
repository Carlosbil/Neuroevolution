# Generación de CSV desde Archivos WAV

Este documento describe el proceso utilizado en el proyecto original [TFM_ParkinsonDetection](https://github.com/mreyp/TFM_ParkinsonDetection) para procesar archivos de audio WAV y convertirlos en archivos CSV para su posterior análisis con redes neuronales.

## Descripción General

El proceso consta de dos etapas principales:
1. **Preprocesamiento de audios**: Normalización de la duración de los archivos WAV
2. **Conversión a CSV**: Transformación de las señales de audio en formato tabular

## 1. Preprocesamiento de Archivos WAV

### Ubicación Original
El código se encuentra en `preparacion/shorten_audios.ipynb`

### Estructura de Datos de Entrada

Los datos se organizaban en dos carpetas principales:
- **Archivos de Control**: `datos/Vowels/Control/A/` - Sujetos sanos
- **Archivos Patológicos**: `datos/Vowels/Patologicas/A/` - Sujetos con Parkinson

Todos los archivos eran grabaciones de la vocal "A" sostenida.

### Proceso de Normalización

#### 1.1. Encontrar la Duración Mínima

```python
def get_wav_duration(file_path):
    """
    Obtiene la duración de un archivo WAV.
    
    Args:
        file_path: Ruta al archivo WAV
    
    Returns:
        duration: Duración en segundos
        sr: Tasa de muestreo
    """
    audio, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration, sr

def truncate_float(float_number, decimal_places):
    """Trunca un número flotante a un número específico de decimales."""
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

# Buscar la duración más corta entre todos los archivos
shortest_duration = float('inf')

for folder_path in [control_folder_path, pathological_folder_path]:
    file_names = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        duration = get_wav_duration(file_path)[0]
        duration = truncate_float(duration, 2)
        if duration < shortest_duration:
            shortest_duration = duration
```

**Razón**: Para garantizar que todos los archivos tengan la misma longitud temporal, necesaria para entrenar modelos de deep learning que requieren entradas de tamaño fijo.

#### 1.2. Acortar Todos los Archivos

```python
def shorten_wav(input_file, output_file, target_duration, sr):
    """
    Acorta un archivo WAV a una duración específica.
    
    Args:
        input_file: Ruta al archivo de entrada
        output_file: Ruta al archivo de salida
        target_duration: Duración objetivo en segundos
        sr: Tasa de muestreo objetivo
    """
    audio, sr = librosa.load(input_file, sr=sr)
    target_frames = int(target_duration * sr)
    shortened_audio = audio[:target_frames]
    sf.write(output_file, shortened_audio, sr)

def shorten_wav_files_24(folder_path, output_folder, shortest_duration):
    """
    Procesa todos los archivos WAV de una carpeta con tasa de muestreo de 24kHz.
    """
    file_names = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    for file_name in file_names:
        input_file = os.path.join(folder_path, file_name)
        output_file = os.path.join(output_folder, file_name.replace('.wav', '_shortened.wav'))
        shorten_wav(input_file, output_file, shortest_duration, 24000)
```

### Tasas de Muestreo

El proyecto generaba archivos con dos tasas de muestreo diferentes:
- **24 kHz**: Para el modelo BigVSAN (generación de audio con GAN)
- **44.1 kHz**: Para otras arquitecturas de clasificación

### Archivos de Salida

Los archivos procesados se guardaban en:
- `datos/control_files_short_24khz/`
- `datos/pathological_files_short_24khz/`
- `datos/control_files_short_44_1khz/`
- `datos/pathological_files_short_44_1khz/`

## 2. Conversión de WAV a CSV

### Ubicación Original
El código también se encuentra en `preparacion/shorten_audios.ipynb`

### Proceso de Conversión

```python
def save_csv(audio_path, output_csv):
    """
    Crea archivos CSV a partir de archivos de audio WAV.
    
    Args:
        audio_path: Ruta a la carpeta con archivos WAV
        output_csv: Ruta a la carpeta de salida para los CSV
    
    Proceso:
        1. Elimina la carpeta de salida si existe
        2. Crea una nueva carpeta de salida
        3. Lee cada archivo WAV
        4. Convierte los datos de audio a DataFrame
        5. Guarda cada archivo como CSV individual
    """
    # Limpiar carpeta de salida
    if os.path.exists(output_csv):
        shutil.rmtree(output_csv)
    os.makedirs(output_csv, exist_ok=True)
    
    # Procesar cada archivo WAV
    input_filenames = [file for file in os.listdir(audio_path) if file.endswith('.wav')]
    for name in input_filenames:
        if name[-3:] != 'wav':
            print('WARNING!! Input File format should be *.wav')
            sys.exit()

        # Leer datos de audio
        sr, data = wavfile.read(os.path.join(audio_path, name))
        
        # Convertir a DataFrame
        wavData = pd.DataFrame(data)
        wavData.columns = ['M']  # Columna única con valores de amplitud
        
        # Guardar como CSV
        wavData.to_csv(os.path.join(output_csv, name[:-4] + ".csv"), mode='w')
```

### Formato del CSV

Cada archivo CSV contiene:
- **Una columna**: `M` (amplitud de la señal)
- **Filas**: Cada fila representa una muestra temporal del audio
- **Valores**: Amplitudes discretas de la señal de audio (típicamente valores enteros entre -32768 y 32767 para audio de 16 bits)

Ejemplo de estructura:
```
,M
0,123
1,456
2,234
3,-123
...
```

## 3. Uso en Modelos de Deep Learning

### Carga de Datos para Entrenamiento

Una vez generados los CSV, se cargaban para entrenar modelos como ResNet, InceptionTime, etc:

```python
# Ejemplo de carga desde CSV
data_train = pd.read_csv('datos/whale_dataset/RightWhaleCalls_train.csv')
y_train = np.array(data_train['label'])
x_train = np.array(data_train.drop(['label'], axis=1))

# Para datos de Parkinson se usaban archivos .npy
x_train = np.load('datos/sets/X_train_40_1e6_N.npy')
y_train = np.load('datos/sets/y_train_40_1e6_N.npy')
```

### Normalización Post-Conversión

Después de cargar los datos CSV, se aplicaban normalizaciones adicionales:

```python
# Normalización con MinMaxScaler
scaler = sklearn.preprocessing.MinMaxScaler()
series_set_scaled = np.zeros_like(series_set)
for i, series in enumerate(series_set):
    series_reshaped = series.reshape(-1, 1)
    scaler.fit(series_reshaped)
    series_set_scaled[i] = scaler.transform(series_reshaped).flatten()
```

## 4. Etiquetado de Datos

Para entrenar modelos supervisados, se asignaban etiquetas:
- **0**: Sujetos de control (sanos)
- **1**: Sujetos patológicos (con Parkinson)

Las etiquetas se organizaban en conjuntos de entrenamiento, validación y test:
```
- train (70%)
- validation (30% del train original)
- test (conjunto separado)
```

## Dependencias Principales

```python
import librosa          # Procesamiento de audio
import soundfile as sf  # Lectura/escritura de audio
import numpy as np      # Operaciones numéricas
import pandas as pd     # Manejo de datos tabulares
from scipy.io import wavfile  # Lectura de archivos WAV
```

## Ventajas de este Enfoque

1. **Uniformidad**: Todos los archivos tienen la misma duración
2. **Compatibilidad**: Los CSV son fáciles de manipular con pandas
3. **Flexibilidad**: Permite trabajar con diferentes tasas de muestreo
4. **Reproducibilidad**: El proceso está bien documentado y es repetible

## Limitaciones

1. **Pérdida de Información**: Se recortan los audios más largos
2. **Tamaño de Archivos**: Los CSV pueden ser más grandes que los WAV comprimidos
3. **Procesamiento Secuencial**: No está optimizado para procesamiento paralelo

## Referencias

- Repositorio Original: [mreyp/TFM_ParkinsonDetection](https://github.com/mreyp/TFM_ParkinsonDetection)
- Paper Asociado: "Time Series Classification of Raw Voice Waveforms for Parkinson's Disease Detection Using Generative Adversarial Network-Driven Data Augmentation" (IEEE OJCS 2024)
