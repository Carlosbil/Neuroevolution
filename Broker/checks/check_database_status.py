#!/usr/bin/env python3
"""
Script para verificar el estado actual de la base de datos de neuroevolución.
Proporciona información detallada sobre poblaciones y modelos.
"""

import os
import sys
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import logging
import argparse
from datetime import datetime, timedelta
import json

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de PostgreSQL
PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_DB = os.environ.get("POSTGRES_DB", "neuroevolution")

def get_db_connection():
    """Crea y retorna una conexión a la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DB,
            connect_timeout=10,
            application_name='neuroevolution_status_checker'
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        raise

def get_database_status(conn, detailed=False):
    """Obtiene el estado completo de la base de datos."""
    try:
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            status = {}
            
            # Información básica
            cursor.execute("SELECT COUNT(*) as count FROM populations")
            status['populations_count'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM models")
            status['models_count'] = cursor.fetchone()['count']
            
            # Información sobre poblaciones
            cursor.execute("""
                SELECT 
                    uuid,
                    created_at,
                    (SELECT COUNT(*) FROM models WHERE population_uuid = p.uuid) as model_count,
                    (SELECT COUNT(*) FROM models WHERE population_uuid = p.uuid AND score > 0) as scored_models,
                    (SELECT MAX(score) FROM models WHERE population_uuid = p.uuid) as max_score,
                    (SELECT AVG(score) FROM models WHERE population_uuid = p.uuid AND score > 0) as avg_score
                FROM populations p
                ORDER BY created_at DESC
            """)
            status['populations'] = cursor.fetchall()
            
            # Estadísticas de modelos
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_models,
                    COUNT(CASE WHEN score > 0 THEN 1 END) as scored_models,
                    MAX(score) as max_score,
                    AVG(CASE WHEN score > 0 THEN score END) as avg_score,
                    MIN(CASE WHEN score > 0 THEN score END) as min_score
                FROM models
            """)
            status['model_stats'] = cursor.fetchone()
            
            # Actividad reciente (últimas 24 horas)
            cursor.execute("""
                SELECT 
                    COUNT(*) as recent_populations
                FROM populations 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            status['recent_populations'] = cursor.fetchone()['recent_populations']
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as recent_models
                FROM models 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            status['recent_models'] = cursor.fetchone()['recent_models']
            
            if detailed:
                # Información detallada de modelos por población
                cursor.execute("""
                    SELECT 
                        p.uuid as population_uuid,
                        p.created_at as population_created,
                        m.model_id,
                        m.score,
                        m.created_at as model_created,
                        m.data
                    FROM populations p
                    LEFT JOIN models m ON p.uuid = m.population_uuid
                    ORDER BY p.created_at DESC, m.score DESC
                """)
                status['detailed_models'] = cursor.fetchall()
            
            return status
            
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado de la base de datos: {e}")
        return None

def print_status(status, detailed=False):
    """Imprime el estado de la base de datos de forma organizada."""
    if not status:
        logger.error("❌ No se pudo obtener el estado de la base de datos")
        return
    
    print("\n" + "="*80)
    print("🔍 ESTADO DE LA BASE DE DATOS DE NEUROEVOLUCIÓN")
    print("="*80)
    
    # Resumen general
    print(f"\n📊 RESUMEN GENERAL")
    print(f"   • Poblaciones totales: {status['populations_count']}")
    print(f"   • Modelos totales: {status['models_count']}")
    print(f"   • Actividad reciente (24h): {status['recent_populations']} poblaciones, {status['recent_models']} modelos")
    
    # Estadísticas de modelos
    if status['model_stats']:
        stats = status['model_stats']
        print(f"\n🎯 ESTADÍSTICAS DE MODELOS")
        print(f"   • Total de modelos: {stats['total_models']}")
        print(f"   • Modelos con score: {stats['scored_models']}")
        if stats['max_score']:
            print(f"   • Score máximo: {stats['max_score']:.4f}")
            print(f"   • Score promedio: {stats['avg_score']:.4f}")
            print(f"   • Score mínimo: {stats['min_score']:.4f}")
        else:
            print(f"   • No hay modelos con score asignado")
    
    # Información de poblaciones
    if status['populations']:
        print(f"\n🧬 POBLACIONES")
        for i, pop in enumerate(status['populations']):
            print(f"   {i+1}. UUID: {pop['uuid']}")
            print(f"      • Creada: {pop['created_at']}")
            print(f"      • Modelos: {pop['model_count']}")
            print(f"      • Con score: {pop['scored_models']}")
            if pop['max_score']:
                print(f"      • Score máximo: {pop['max_score']:.4f}")
                print(f"      • Score promedio: {pop['avg_score']:.4f}")
            print()
    else:
        print(f"\n🧬 POBLACIONES")
        print(f"   No hay poblaciones registradas")
    
    # Información detallada si se solicita
    if detailed and 'detailed_models' in status:
        print(f"\n🔬 DETALLE DE MODELOS")
        current_pop = None
        for model in status['detailed_models']:
            if model['population_uuid'] != current_pop:
                current_pop = model['population_uuid']
                print(f"\n   📁 Población: {current_pop}")
                print(f"      Creada: {model['population_created']}")
            
            if model['model_id']:
                print(f"      • {model['model_id']}")
                print(f"        Score: {model['score'] if model['score'] else 'Sin score'}")
                print(f"        Creado: {model['model_created']}")
                if detailed and model['data']:
                    # Mostrar solo información básica del modelo
                    data = model['data']
                    if isinstance(data, dict):
                        print(f"        Capas: {len(data.get('layers', []))}")
                        print(f"        Parámetros: {data.get('total_params', 'N/A')}")
    
    print("="*80)

def clean_database(conn):
    """Limpia la base de datos eliminando todos los registros."""
    try:
        with conn.cursor() as cursor:
            logger.info("🧹 Limpiando base de datos...")
            cursor.execute("DELETE FROM models")
            cursor.execute("DELETE FROM populations")
            conn.commit()
            logger.info("✅ Base de datos limpiada exitosamente")
    except Exception as e:
        logger.error(f"❌ Error limpiando base de datos: {e}")
        conn.rollback()
        raise

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Verificador de estado de la base de datos de neuroevolución')
    parser.add_argument('--detailed', '-d', action='store_true', help='Mostrar información detallada')
    parser.add_argument('--json', '-j', action='store_true', help='Salida en formato JSON')
    parser.add_argument('--clean', action='store_true', help='Limpiar la base de datos (¡CUIDADO!)')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS', help='Monitorear continuamente cada N segundos')
    
    args = parser.parse_args()
    
    try:
        # Conectar a la base de datos
        conn = get_db_connection()
        logger.info("✅ Conectado a la base de datos")
        
        # Limpiar base de datos si se solicita
        if args.clean:
            response = input("⚠️ ¿Estás seguro de que quieres limpiar la base de datos? (escribe 'SI' para confirmar): ")
            if response == 'SI':
                clean_database(conn)
                return 0
            else:
                logger.info("❌ Limpieza cancelada")
                return 0
        
        # Monitoreo continuo
        if args.watch:
            logger.info(f"👁️ Iniciando monitoreo continuo cada {args.watch} segundos (Ctrl+C para salir)")
            try:
                while True:
                    os.system('cls' if os.name == 'nt' else 'clear')  # Limpiar pantalla
                    status = get_database_status(conn, args.detailed)
                    if args.json:
                        # Convertir a JSON serializable
                        json_status = {}
                        for key, value in status.items():
                            if isinstance(value, list):
                                json_status[key] = [dict(item) if hasattr(item, 'keys') else item for item in value]
                            elif hasattr(value, 'keys'):
                                json_status[key] = dict(value)
                            else:
                                json_status[key] = value
                        print(json.dumps(json_status, indent=2, default=str))
                    else:
                        print(f"🕐 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print_status(status, args.detailed)
                    
                    import time
                    time.sleep(args.watch)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Monitoreo detenido por el usuario")
        else:
            # Verificación única
            status = get_database_status(conn, args.detailed)
            
            if args.json:
                # Convertir a JSON serializable
                json_status = {}
                for key, value in status.items():
                    if isinstance(value, list):
                        json_status[key] = [dict(item) if hasattr(item, 'keys') else item for item in value]
                    elif hasattr(value, 'keys'):
                        json_status[key] = dict(value)
                    else:
                        json_status[key] = value
                print(json.dumps(json_status, indent=2, default=str))
            else:
                print_status(status, args.detailed)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    finally:
        if 'conn' in locals():
            conn.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
