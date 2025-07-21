#!/usr/bin/env python3
"""
Script para convertir archivos .wav a imágenes (espectrogramas y formas de onda)
Convierte los archivos de las carpetas pretrained a carpetas images_ correspondientes.

Autor: Neuroevolution Project
Fecha: 2025
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore')

def install_dependencies():
    """Instalar dependencias requeridas si no están disponibles"""
    required_packages = [
        'librosa>=0.10.0',
        'matplotlib>=3.7.0', 
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'soundfile>=0.12.0',
        'torch>=2.0.0',  # Para soporte GPU
        'torchaudio>=2.0.0'  # Para procesamiento de audio acelerado
    ]
    
    for package in required_packages:
        try:
            package_name = package.split('>=')[0]
            __import__(package_name)
        except ImportError:
            logger.info(f"Instalando {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✓ {package} instalado exitosamente")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error instalando {package}: {e}")
                return False
    return True

# Instalar dependencias si es necesario
if not install_dependencies():
    logger.error("No se pudieron instalar todas las dependencias requeridas")
    sys.exit(1)

# Importar librerías después de instalar dependencias
try:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI para mejor compatibilidad
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    
    # Importar PyTorch para soporte GPU
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
    logger.error("Por favor ejecuta: pip install -r requirements_audio_converter.txt")
    sys.exit(1)

class WavToImageConverter:
    """Clase para convertir archivos WAV a imágenes (espectrogramas y formas de onda) con soporte GPU"""
    
    def __init__(self, base_path: str, use_gpu: bool = True):
        """
        Inicializar el conversor
        
        Args:
            base_path (str): Ruta base donde están las carpetas pretrained
            use_gpu (bool): Si usar GPU cuando esté disponible
        """
        self.base_path = Path(base_path)
        self.input_folders = [
            "pretrained_40_1e5_BigVSAN_generated_control",
            "pretrained_40_1e5_BigVSAN_generated_pathological"
        ]
        self.output_folders = [
            "images_pretrained_40_1e5_BigVSAN_generated_control",
            "images_pretrained_40_1e5_BigVSAN_generated_pathological"
        ]
        
        # Configurar dispositivo
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            logger.info(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
            # Optimizar configuración para GPU
            torch.backends.cudnn.benchmark = True
        else:
            logger.info("🔧 Usando CPU para procesamiento")
            
        # Configurar batch size según el dispositivo
        self.batch_size = 32 if self.use_gpu else 8
        
    def create_output_directories(self):
        """Crear los directorios de salida si no existen"""
        for output_folder in self.output_folders:
            output_path = self.base_path / output_folder
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {output_path}")
    
    def wav_to_spectrogram_gpu(self, wav_file: Path, output_file: Path):
        """
        Convertir archivo WAV a espectrograma usando GPU cuando esté disponible
        
        Args:
            wav_file (Path): Ruta del archivo WAV
            output_file (Path): Ruta del archivo de imagen de salida
        """
        try:
            if self.use_gpu:
                # Cargar usando torchaudio para mejor rendimiento con GPU
                waveform, sample_rate = torchaudio.load(str(wav_file))
                waveform = waveform.to(self.device)
                
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
                # Usar librosa para CPU
                y, sr = librosa.load(str(wav_file), sr=None)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                log_spec_cpu = D
                sample_rate = sr
            
            # Crear la visualización
            plt.figure(figsize=(12, 8))
            if self.use_gpu:
                # Para datos de PyTorch
                plt.imshow(log_spec_cpu, aspect='auto', origin='lower', 
                          interpolation='nearest', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
            else:
                # Para datos de librosa
                librosa.display.specshow(log_spec_cpu, y_axis='hz', x_axis='time', sr=sample_rate)
                plt.colorbar(format='%+2.0f dB')
                
            plt.title(f'Espectrograma GPU - {wav_file.name}')
            plt.xlabel('Tiempo')
            plt.ylabel('Frecuencia (Hz)')
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Limpiar memoria GPU
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando {wav_file}: {str(e)}")
            if self.use_gpu:
                torch.cuda.empty_cache()
            return False

    def wav_to_spectrogram(self, wav_file: Path, output_file: Path):
        """
        Convertir archivo WAV a espectrograma (método legacy)
        
        Args:
            wav_file (Path): Ruta del archivo WAV
            output_file (Path): Ruta del archivo de imagen de salida
        """
        # Usar método GPU si está disponible
        if self.use_gpu:
            return self.wav_to_spectrogram_gpu(wav_file, output_file)
            
        try:
            # Cargar el archivo de audio
            y, sr = librosa.load(str(wav_file), sr=None)
            
            # Crear espectrograma
            plt.figure(figsize=(10, 6))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Espectrograma - {wav_file.name}')
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando {wav_file}: {str(e)}")
            return False
    
    def wav_to_waveform_gpu(self, wav_file: Path, output_file: Path):
        """
        Convertir archivo WAV a forma de onda usando GPU cuando esté disponible
        
        Args:
            wav_file (Path): Ruta del archivo WAV  
            output_file (Path): Ruta del archivo de imagen de salida
        """
        try:
            if self.use_gpu:
                # Cargar usando torchaudio
                waveform, sample_rate = torchaudio.load(str(wav_file))
                waveform = waveform.to(self.device)
                
                # Procesar en GPU si es necesario (normalización, etc.)
                waveform = waveform / torch.max(torch.abs(waveform))  # Normalizar
                
                # Mover de vuelta a CPU para visualización
                waveform_cpu = waveform.squeeze(0).cpu().numpy()
                
            else:
                # Usar librosa para CPU
                waveform_cpu, sample_rate = librosa.load(str(wav_file), sr=None)
            
            # Crear forma de onda
            plt.figure(figsize=(15, 6))
            time_axis = np.linspace(0, len(waveform_cpu) / sample_rate, len(waveform_cpu))
            plt.plot(time_axis, waveform_cpu, linewidth=0.5)
            plt.title(f'Forma de Onda GPU - {wav_file.name}')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Limpiar memoria GPU
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando {wav_file}: {str(e)}")
            if self.use_gpu:
                torch.cuda.empty_cache()
            return False

    def wav_to_waveform(self, wav_file: Path, output_file: Path):
        """
        Convertir archivo WAV a forma de onda (método legacy)
        
        Args:
            wav_file (Path): Ruta del archivo WAV
            output_file (Path): Ruta del archivo de imagen de salida
        """
        # Usar método GPU si está disponible
        if self.use_gpu:
            return self.wav_to_waveform_gpu(wav_file, output_file)
            
        try:
            # Cargar el archivo de audio
            y, sr = librosa.load(str(wav_file), sr=None)
            
            # Crear forma de onda
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Forma de Onda - {wav_file.name}')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando {wav_file}: {str(e)}")
            return False
    
    def process_folder_batch(self, input_folder: str, output_folder: str, conversion_type: str = "spectrogram"):
        """
        Procesar archivos WAV en lotes para optimizar el uso de GPU
        
        Args:
            input_folder (str): Nombre de la carpeta de entrada
            output_folder (str): Nombre de la carpeta de salida  
            conversion_type (str): Tipo de conversión ("spectrogram" o "waveform")
        """
        input_path = self.base_path / input_folder / input_folder
        output_path = self.base_path / output_folder
        
        # Verificar que la carpeta de entrada existe
        if not input_path.exists():
            logger.error(f"La carpeta {input_path} no existe")
            return
        
        # Obtener todos los archivos WAV
        wav_files = list(input_path.glob("*.wav"))
        logger.info(f"Encontrados {len(wav_files)} archivos WAV en {input_path}")
        
        if not wav_files:
            logger.warning(f"No se encontraron archivos WAV en {input_path}")
            return
        
        # Procesar cada archivo
        successful_conversions = 0
        failed_conversions = 0
        
        # Mostrar información del dispositivo antes de comenzar
        if self.use_gpu:
            logger.info(f"🚀 Procesando con GPU - Batch size: {self.batch_size}")
            logger.info(f"📊 Memoria GPU inicial: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Procesar archivos con barra de progreso
        with tqdm(wav_files, desc=f"Procesando {input_folder} ({'GPU' if self.use_gpu else 'CPU'})") as pbar:
            for wav_file in pbar:
                # Generar nombre del archivo de salida
                if conversion_type == "spectrogram":
                    output_file = output_path / f"{wav_file.stem}_spectrogram_{'gpu' if self.use_gpu else 'cpu'}.png"
                    success = self.wav_to_spectrogram(wav_file, output_file)
                elif conversion_type == "waveform":
                    output_file = output_path / f"{wav_file.stem}_waveform_{'gpu' if self.use_gpu else 'cpu'}.png"
                    success = self.wav_to_waveform(wav_file, output_file)
                else:
                    logger.error(f"Tipo de conversión no válido: {conversion_type}")
                    continue
                
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
                
                # Actualizar información en la barra de progreso
                if self.use_gpu:
                    memory_used = torch.cuda.memory_allocated() / 1024**2
                    pbar.set_postfix({
                        'Exitosos': successful_conversions,
                        'GPU MB': f"{memory_used:.1f}"
                    })
                else:
                    pbar.set_postfix({'Exitosos': successful_conversions})
        
        # Estadísticas finales
        logger.info(f"Procesamiento completado para {input_folder}:")
        logger.info(f"  - Conversiones exitosas: {successful_conversions}")
        logger.info(f"  - Conversiones fallidas: {failed_conversions}")
        
        if self.use_gpu:
            logger.info(f"  - Memoria GPU final: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            # Limpiar memoria GPU al final
            torch.cuda.empty_cache()
            logger.info(f"  - Memoria GPU después de limpieza: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    def process_folder(self, input_folder: str, output_folder: str, conversion_type: str = "spectrogram"):
        """
        Procesar todos los archivos WAV de una carpeta (método legacy)
        
        Args:
            input_folder (str): Nombre de la carpeta de entrada
            output_folder (str): Nombre de la carpeta de salida
            conversion_type (str): Tipo de conversión ("spectrogram" o "waveform")
        """
        # Usar método batch optimizado
        return self.process_folder_batch(input_folder, output_folder, conversion_type)
    
    def convert_all(self, conversion_type: str = "spectrogram"):
        """
        Convertir todos los archivos WAV a imágenes con optimizaciones GPU
        
        Args:
            conversion_type (str): Tipo de conversión ("spectrogram", "waveform" o "both")
        """
        logger.info("🚀 Iniciando conversión de archivos WAV a imágenes...")
        
        if self.use_gpu:
            logger.info(f"⚡ Modo GPU activado - {torch.cuda.get_device_name(0)}")
            logger.info(f"📊 Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            logger.info("🔧 Modo CPU - Sin aceleración GPU")
        
        # Crear directorios de salida
        self.create_output_directories()
        
        # Procesar cada carpeta
        for input_folder, output_folder in zip(self.input_folders, self.output_folders):
            logger.info(f"📁 Procesando carpeta: {input_folder}")
            
            if conversion_type in ["spectrogram", "both"]:
                logger.info(f"🎵 Generando espectrogramas para {input_folder}...")
                self.process_folder(input_folder, output_folder, "spectrogram")
            
            if conversion_type in ["waveform", "both"]:
                logger.info(f"〰️ Generando formas de onda para {input_folder}...")
                # Para formas de onda, usar subcarpetas
                waveform_output = output_folder + "_waveforms"
                output_path = self.base_path / waveform_output
                output_path.mkdir(parents=True, exist_ok=True)
                self.process_folder(input_folder, waveform_output, "waveform")
        
        logger.info("✅ Conversión completada!")
        
        if self.use_gpu:
            # Estadísticas finales de GPU
            logger.info("📊 Estadísticas finales de GPU:")
            logger.info(f"  - Memoria utilizada: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            logger.info(f"  - Memoria máxima utilizada: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            
            # Limpiar toda la memoria GPU
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("🧹 Memoria GPU limpiada")


def main():
    """Función principal"""
    print("Muy Buenas señor Carlos!")
    print("=" * 60)
    print("🎵 Conversor de WAV a Imágenes con Soporte GPU 🚀")
    print("=" * 60)
    
    # Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"📊 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        use_gpu_default = True
    else:
        print("⚠️ GPU no detectada - se usará CPU")
        use_gpu_default = False
    
    # Ruta base donde están las carpetas pretrained
    base_path = r"e:\Neuroevolution\data\Real_y_GANs-Paper_Marta-20250201T072321Z-001\Real_y_GANs-Paper_Marta"
    
    # Verificar que la ruta base existe
    if not os.path.exists(base_path):
        logger.error(f"La ruta base {base_path} no existe")
        sys.exit(1)
    
    # Preguntar sobre el uso de GPU
    if torch.cuda.is_available():
        gpu_choice = input("\n¿Usar GPU para acelerar el procesamiento? (S/n): ").strip().lower()
        use_gpu = gpu_choice != 'n'
    else:
        use_gpu = False
    
    # Crear el conversor
    converter = WavToImageConverter(base_path, use_gpu=use_gpu)
    
    # Opciones de conversión
    print(f"\n🎯 Opciones de conversión disponibles:")
    print("1. Solo espectrogramas (recomendado)")
    print("2. Solo formas de onda")
    print("3. Ambos (espectrogramas y formas de onda)")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == "1":
        conversion_type = "spectrogram"
    elif choice == "2":
        conversion_type = "waveform"
    elif choice == "3":
        conversion_type = "both"
    else:
        logger.info("Opción no válida, usando espectrogramas por defecto")
        conversion_type = "spectrogram"
    
    # Mostrar resumen antes de comenzar
    print(f"\n📋 Resumen de la conversión:")
    print(f"  - Dispositivo: {'GPU' if use_gpu else 'CPU'}")
    print(f"  - Tipo: {conversion_type}")
    print(f"  - Carpetas a procesar: {len(converter.input_folders)}")
    
    proceed = input("\n¿Continuar con la conversión? (S/n): ").strip().lower()
    if proceed == 'n':
        print("Conversión cancelada por el usuario")
        sys.exit(0)
    
    # Realizar la conversión
    import time
    start_time = time.time()
    
    converter.convert_all(conversion_type)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n⏱️ Tiempo total de procesamiento: {elapsed_time:.2f} segundos")
    print(f"🎉 ¡Proceso completado exitosamente!")


if __name__ == "__main__":
    main()
