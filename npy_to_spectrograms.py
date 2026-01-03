#!/usr/bin/env python3
"""
Script para convertir archivos .npy (datos de audio) a espectrogramas
Procesa los archivos de la carpeta folds_5/files_all_real_syn_n

Autor: Neuroevolution Project
Fecha: 2025
"""

import os
import sys
from pathlib import Path
import logging
import warnings
import numpy as np
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore')

# Importar librerías necesarias
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    import matplotlib.pyplot as plt
    import torch
    import torchaudio
    
    # Configurar dispositivo (GPU si está disponible)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA versión: {torch.version.cuda}")
        logger.info(f"✓ Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        device = torch.device('cpu')
        logger.info("⚠️ GPU no detectada, usando CPU")
        
except ImportError as e:
    logger.error(f"Error importando dependencias: {e}")
    logger.error("Por favor ejecuta: pip install matplotlib torch torchaudio tqdm numpy")
    sys.exit(1)


class NpyToSpectrogramConverter:
    """Clase para convertir archivos .npy a espectrogramas con soporte GPU"""
    
    def __init__(self, input_folder: str, output_folder: str, use_gpu: bool = True, sample_rate: int = 24000):
        """
        Inicializar el conversor
        
        Args:
            input_folder (str): Ruta de la carpeta con archivos .npy
            output_folder (str): Ruta donde guardar los espectrogramas
            use_gpu (bool): Si usar GPU cuando esté disponible
            sample_rate (int): Frecuencia de muestreo del audio (24kHz por defecto)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sample_rate = sample_rate
        
        # Configurar dispositivo
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            logger.info(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            logger.info("🔧 Usando CPU para procesamiento")
        
        # Crear directorio de salida si no existe
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Directorio de salida: {self.output_folder}")
    
    def npy_to_spectrogram(self, data: np.ndarray, output_file: Path, sample_index: int, label: int):
        """
        Convertir datos de audio (numpy array) a espectrograma
        
        Args:
            data (np.ndarray): Datos de audio en formato numpy (1D array)
            output_file (Path): Ruta del archivo de imagen de salida
            sample_index (int): Índice de la muestra
            label (int): Etiqueta (0=control, 1=patológico)
        """
        try:
            if self.use_gpu:
                # Convertir a tensor de PyTorch y mover a GPU
                waveform = torch.from_numpy(data).float().to(self.device)
                
                # Asegurar que el waveform tenga la forma correcta
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                
                # Configurar transformación STFT optimizada para GPU
                n_fft = 2048
                hop_length = 512
                
                # Aplicar STFT usando PyTorch (GPU acelerado)
                stft = torch.stft(
                    waveform.squeeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=torch.hann_window(n_fft).to(self.device),
                    return_complex=True
                )
                
                # Convertir a espectrograma de potencia
                magnitude = torch.abs(stft)
                power_spec = magnitude ** 2
                
                # Convertir a escala dB
                log_spec = torch.log10(power_spec + 1e-8) * 10
                
                # Mover de vuelta a CPU para visualización
                log_spec_cpu = log_spec.cpu().numpy()
                
            else:
                # Usar numpy/scipy para CPU
                from scipy import signal
                
                # Calcular STFT
                f, t, Zxx = signal.stft(data, fs=self.sample_rate, nperseg=2048, noverlap=1536)
                
                # Convertir a espectrograma de potencia en dB
                magnitude = np.abs(Zxx)
                power_spec = magnitude ** 2
                log_spec_cpu = 10 * np.log10(power_spec + 1e-8)
            
            # Crear la visualización
            plt.figure(figsize=(12, 8))
            plt.imshow(log_spec_cpu, aspect='auto', origin='lower', 
                      interpolation='nearest', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            
            label_text = "Control" if label == 0 else "Patológico"
            plt.title(f'Espectrograma - Sample {sample_index} ({label_text})')
            plt.xlabel('Tiempo')
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Limpiar memoria GPU
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando muestra {sample_index}: {str(e)}")
            if self.use_gpu:
                torch.cuda.empty_cache()
            return False
    
    def process_npy_file(self, npy_file: Path, label_file: Path, subset: str):
        """
        Procesar un archivo .npy y generar espectrogramas para cada muestra
        
        Args:
            npy_file (Path): Ruta del archivo .npy con datos
            label_file (Path): Ruta del archivo .npy con etiquetas
            subset (str): Nombre del subset (train, val, test)
        """
        try:
            # Cargar datos y etiquetas
            logger.info(f"📂 Cargando {npy_file.name}...")
            data = np.load(str(npy_file))
            labels = np.load(str(label_file))
            
            logger.info(f"   Shape de datos: {data.shape}")
            logger.info(f"   Shape de etiquetas: {labels.shape}")
            logger.info(f"   Etiquetas únicas: {np.unique(labels, return_counts=True)}")
            
            # Crear subcarpeta para este subset
            subset_folder = self.output_folder / subset
            subset_folder.mkdir(parents=True, exist_ok=True)
            
            # Procesar cada muestra
            successful = 0
            failed = 0
            
            # Generar espectrogramas para TODAS las muestras
            indices = np.arange(len(data))
            
            logger.info(f"   Generando {len(indices)} espectrogramas...")
            
            with tqdm(indices, desc=f"Procesando {subset}") as pbar:
                for idx in pbar:
                    sample = data[idx]
                    label = int(labels[idx])
                    
                    # Generar nombre del archivo
                    label_name = "control" if label == 0 else "pathological"
                    output_file = subset_folder / f"{subset}_sample_{idx:05d}_{label_name}.png"
                    
                    # Generar espectrograma
                    if self.npy_to_spectrogram(sample, output_file, idx, label):
                        successful += 1
                    else:
                        failed += 1
                    
                    # Actualizar barra de progreso
                    if self.use_gpu:
                        memory_used = torch.cuda.memory_allocated() / 1024**2
                        pbar.set_postfix({
                            'Exitosos': successful,
                            'GPU MB': f"{memory_used:.1f}"
                        })
                    else:
                        pbar.set_postfix({'Exitosos': successful})
            
            logger.info(f"✅ {subset}: {successful} exitosos, {failed} fallidos")
            
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return successful, failed
            
        except Exception as e:
            logger.error(f"Error procesando archivo {npy_file}: {str(e)}")
            return 0, 0
    
    def convert_all_folds(self):
        """
        Convertir todos los folds de files_all_real_syn_n a espectrogramas
        """
        logger.info("🚀 Iniciando conversión de .npy a espectrogramas...")
        
        if self.use_gpu:
            logger.info(f"⚡ Modo GPU activado - {torch.cuda.get_device_name(0)}")
        else:
            logger.info("🔧 Modo CPU")
        
        # Buscar todos los archivos .npy de datos (X_*)
        x_files = sorted(self.input_folder.glob("X_*.npy"))
        
        if not x_files:
            logger.error(f"No se encontraron archivos .npy en {self.input_folder}")
            return
        
        logger.info(f"📊 Encontrados {len(x_files)} archivos de datos")
        
        total_successful = 0
        total_failed = 0
        
        # Procesar cada archivo
        for x_file in x_files:
            # Obtener el archivo de etiquetas correspondiente
            y_file = self.input_folder / x_file.name.replace('X_', 'y_')
            
            if not y_file.exists():
                logger.warning(f"No se encontró archivo de etiquetas para {x_file.name}")
                continue
            
            # Determinar el subset (train, val, test)
            if 'train' in x_file.name:
                subset = 'train'
            elif 'val' in x_file.name:
                subset = 'val'
            elif 'test' in x_file.name:
                subset = 'test'
            else:
                subset = 'other'
            
            # Extraer número de fold
            fold_num = x_file.name.split('fold_')[-1].split('.')[0]
            subset_name = f"{subset}_fold_{fold_num}"
            
            # Procesar el archivo
            successful, failed = self.process_npy_file(x_file, y_file, subset_name)
            total_successful += successful
            total_failed += failed
        
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN FINAL:")
        logger.info(f"   ✅ Total exitosos: {total_successful}")
        logger.info(f"   ❌ Total fallidos: {total_failed}")
        logger.info(f"   📁 Espectrogramas guardados en: {self.output_folder}")
        
        if self.use_gpu:
            logger.info(f"\n📊 Estadísticas GPU:")
            logger.info(f"   Memoria máxima utilizada: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        logger.info("="*60)


def main():
    """Función principal"""
    print("="*60)
    print("🎵 Conversor de NPY a Espectrogramas 🚀")
    print("="*60)
    
    # Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print("⚠️ GPU no detectada - se usará CPU")
        use_gpu = False
    
    # Configurar rutas
    input_folder = r"e:\Neuroevolution\data\sets\folds_5\files_syn_40_1e5_N"
    output_folder = r"e:\Neuroevolution\data\sets\folds_5\files_syn_40_1e5_N_spectrograms"
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(input_folder):
        logger.error(f"La carpeta {input_folder} no existe")
        sys.exit(1)
    
    print(f"\n📋 Configuración:")
    print(f"   - Entrada: {input_folder}")
    print(f"   - Salida: {output_folder}")
    print(f"   - Dispositivo: {'GPU' if use_gpu else 'CPU'}")
    print(f"   - Sample rate: 24000 Hz")
    print(f"\n🚀 Iniciando conversión automáticamente...")
    
    # Crear el conversor
    converter = NpyToSpectrogramConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        use_gpu=use_gpu,
        sample_rate=24000
    )
    
    # Realizar la conversión
    import time
    start_time = time.time()
    
    converter.convert_all_folds()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n⏱️ Tiempo total: {elapsed_time:.2f} segundos")
    print(f"🎉 ¡Proceso completado!")


if __name__ == "__main__":
    main()
