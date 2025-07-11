#!/usr/bin/env python3
"""
Script para verificar el estado del directorio de modelos y ayudar con el debug.
"""

import os
import sys
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
import argparse

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración del path de storage
STORAGE_PATH = os.environ.get("BROKER_STORAGE_PATH", os.path.join(os.path.dirname(__file__), 'models'))

def check_storage_directory():
    """Verifica el estado del directorio de storage."""
    logger.info(f"📁 Verificando directorio de storage: {STORAGE_PATH}")
    
    if not os.path.exists(STORAGE_PATH):
        logger.error(f"❌ El directorio de storage no existe: {STORAGE_PATH}")
        return False
    
    # Listar todos los archivos
    try:
        files = os.listdir(STORAGE_PATH)
        logger.info(f"📂 Archivos encontrados en el directorio: {len(files)}")
        
        if not files:
            logger.warning("⚠️ El directorio de storage está vacío")
            return True
        
        # Analizar cada archivo
        json_files = []
        other_files = []
        
        for file in files:
            file_path = os.path.join(STORAGE_PATH, file)
            if file.endswith('.json'):
                json_files.append(file)
            else:
                other_files.append(file)
        
        logger.info(f"📄 Archivos JSON: {len(json_files)}")
        logger.info(f"📄 Otros archivos: {len(other_files)}")
        
        # Mostrar archivos JSON
        for json_file in json_files:
            logger.info(f"  • {json_file}")
            
        # Mostrar otros archivos
        for other_file in other_files:
            logger.info(f"  • {other_file} (no JSON)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error leyendo directorio de storage: {e}")
        return False

def analyze_json_file(filename):
    """Analiza un archivo JSON específico."""
    file_path = os.path.join(STORAGE_PATH, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"🔍 Analizando archivo: {filename}")
        logger.info(f"📊 Tipo de datos: {type(data)}")
        
        if isinstance(data, dict):
            logger.info(f"📋 Claves principales: {list(data.keys())}")
            logger.info(f"📊 Número de elementos: {len(data)}")
            
            # Analizar cada elemento
            for key, value in data.items():
                logger.info(f"  • {key}: {type(value)}")
                if isinstance(value, dict):
                    logger.info(f"    - Subclaves: {list(value.keys())}")
                elif isinstance(value, list):
                    logger.info(f"    - Elementos en lista: {len(value)}")
                    if value:
                        logger.info(f"    - Tipo de primer elemento: {type(value[0])}")
        
        elif isinstance(data, list):
            logger.info(f"📊 Elementos en lista: {len(data)}")
            if data:
                logger.info(f"📋 Tipo de primer elemento: {type(data[0])}")
                if isinstance(data[0], dict):
                    logger.info(f"📋 Claves del primer elemento: {list(data[0].keys())}")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decodificando JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error analizando archivo: {e}")
        return False

def create_test_model():
    """Crea un archivo de modelo de prueba."""
    test_uuid = "test-model-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    test_file = f"{test_uuid}.json"
    test_path = os.path.join(STORAGE_PATH, test_file)
    
    # Crear directorio si no existe
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    # Crear modelo de prueba
    test_data = {
        "model_1": {
            "layers": [
                {"type": "Conv2D", "filters": 32, "kernel_size": 3},
                {"type": "MaxPooling2D", "pool_size": 2},
                {"type": "Conv2D", "filters": 64, "kernel_size": 3},
                {"type": "MaxPooling2D", "pool_size": 2},
                {"type": "Flatten"},
                {"type": "Dense", "units": 128, "activation": "relu"},
                {"type": "Dense", "units": 10, "activation": "softmax"}
            ],
            "total_params": 1000000,
            "created_at": datetime.now().isoformat()
        },
        "model_2": {
            "layers": [
                {"type": "Conv2D", "filters": 16, "kernel_size": 5},
                {"type": "MaxPooling2D", "pool_size": 2},
                {"type": "Conv2D", "filters": 32, "kernel_size": 3},
                {"type": "GlobalAveragePooling2D"},
                {"type": "Dense", "units": 10, "activation": "softmax"}
            ],
            "total_params": 500000,
            "created_at": datetime.now().isoformat()
        }
    }
    
    try:
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"✅ Archivo de prueba creado: {test_file}")
        logger.info(f"📁 Ruta completa: {test_path}")
        logger.info(f"🧪 UUID de prueba: {test_uuid}")
        
        return test_uuid
        
    except Exception as e:
        logger.error(f"❌ Error creando archivo de prueba: {e}")
        return None

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Verificador del directorio de modelos')
    parser.add_argument('--analyze', '-a', type=str, help='Analizar un archivo JSON específico')
    parser.add_argument('--create-test', '-c', action='store_true', help='Crear un archivo de modelo de prueba')
    parser.add_argument('--list-all', '-l', action='store_true', help='Listar y analizar todos los archivos JSON')
    
    args = parser.parse_args()
    
    logger.info("🔍 Iniciando verificación del directorio de modelos")
    
    # Verificar directorio de storage
    if not check_storage_directory():
        return 1
    
    # Crear archivo de prueba si se solicita
    if args.create_test:
        test_uuid = create_test_model()
        if test_uuid:
            logger.info(f"💡 Puedes usar este UUID para pruebas: {test_uuid}")
    
    # Analizar archivo específico
    if args.analyze:
        if not args.analyze.endswith('.json'):
            args.analyze += '.json'
        analyze_json_file(args.analyze)
    
    # Listar y analizar todos los archivos JSON
    if args.list_all:
        try:
            files = os.listdir(STORAGE_PATH)
            json_files = [f for f in files if f.endswith('.json')]
            
            if not json_files:
                logger.info("📂 No hay archivos JSON para analizar")
            else:
                logger.info(f"📂 Analizando {len(json_files)} archivos JSON...")
                for json_file in json_files:
                    logger.info(f"\n{'='*60}")
                    analyze_json_file(json_file)
                    
        except Exception as e:
            logger.error(f"❌ Error listando archivos: {e}")
    
    # Si no se especifica ninguna acción, solo mostrar el estado
    if not any([args.analyze, args.create_test, args.list_all]):
        logger.info("💡 Usa --help para ver las opciones disponibles")
        logger.info("💡 Ejemplos:")
        logger.info("  python check_models_directory.py --list-all")
        logger.info("  python check_models_directory.py --analyze <uuid>")
        logger.info("  python check_models_directory.py --create-test")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
