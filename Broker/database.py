import os
import json
import psycopg2
from psycopg2.extras import Json, DictCursor
import uuid
from utils import logger

# Environment variables for PostgreSQL connection
PG_HOST = os.environ.get("POSTGRES_HOST", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_DB = os.environ.get("POSTGRES_DB", "neuroevolution")


def get_db_connection():
    """
    Create and return a connection to the PostgreSQL database.
    
    :return: PostgreSQL database connection
    :rtype: psycopg2.extensions.connection
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DB
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


def init_db():
    """
    Initialize the database by creating necessary tables if they don't exist.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create populations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS populations (
            uuid VARCHAR(36) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create models table with JSONB for flexible schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            population_uuid VARCHAR(36) REFERENCES populations(uuid),
            model_id VARCHAR(50) NOT NULL,
            data JSONB NOT NULL,
            score FLOAT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(population_uuid, model_id)
        )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def save_population(population_uuid):
    """
    Save a new population entry to the database.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :return: True if successful, False otherwise
    :rtype: bool
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO populations (uuid) VALUES (%s) ON CONFLICT (uuid) DO NOTHING",
            (population_uuid,)
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving population {population_uuid}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def save_model(population_uuid, model_id, model_data, score=None):
    """
    Save a model to the database.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :param model_id: ID of the model within the population
    :type model_id: str
    :param model_data: Model configuration data
    :type model_data: dict
    :param score: Model evaluation score (optional)
    :type score: float
    :return: True if successful, False otherwise
    :rtype: bool
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure population exists
        save_population(population_uuid)
        
        # If score is provided, include it in the model data
        if score is not None:
            model_data["score"] = score
        
        cursor.execute("""
        INSERT INTO models (population_uuid, model_id, data, score)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (population_uuid, model_id) DO UPDATE
        SET data = %s, score = %s
        """, (population_uuid, model_id, Json(model_data), 
              model_data.get("score", 0), Json(model_data), model_data.get("score", 0)))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving model {model_id} for population {population_uuid}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def update_model_score(population_uuid, model_id, score):
    """
    Update the score of a specific model.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :param model_id: ID of the model within the population
    :type model_id: str
    :param score: Model evaluation score
    :type score: float
    :return: True if successful, False otherwise
    :rtype: bool
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the score in both the score column and the JSONB data
        cursor.execute("""
        UPDATE models
        SET score = %s, data = jsonb_set(data, '{score}', %s::jsonb)
        WHERE population_uuid = %s AND model_id = %s
        """, (score, json.dumps(score), population_uuid, model_id))
        
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating score for model {model_id} in population {population_uuid}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def get_population(population_uuid):
    """
    Retrieve all models for a specific population.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :return: Dictionary of models with model_id as keys
    :rtype: dict
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(DictCursor)
        
        cursor.execute("""
        SELECT model_id, data, score
        FROM models
        WHERE population_uuid = %s
        """, (population_uuid,))
        
        models = {}
        for row in cursor.fetchall():
            models[row['model_id']] = row['data']
        
        return models
    except Exception as e:
        logger.error(f"Error retrieving population {population_uuid}: {e}")
        return {}
    finally:
        if conn:
            conn.close()


def population_exists(population_uuid):
    """
    Check if a population exists in the database.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :return: True if the population exists, False otherwise
    :rtype: bool
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT EXISTS(SELECT 1 FROM models WHERE population_uuid = %s)
        """, (population_uuid,))
        
        return cursor.fetchone()[0]
    except Exception as e:
        logger.error(f"Error checking if population {population_uuid} exists: {e}")
        return False
    finally:
        if conn:
            conn.close()


def import_json_models():
    """
    Import existing JSON model files into the PostgreSQL database.
    """
    import glob
    from utils import get_storage_path
    
    models_path = get_storage_path()
    json_files = glob.glob(os.path.join(models_path, "*.json"))
    
    for json_file in json_files:
        try:
            # Extract UUID from filename
            filename = os.path.basename(json_file)
            population_uuid = os.path.splitext(filename)[0]
            
            # Skip files with _best50percent suffix
            if "_best50percent" in population_uuid:
                continue
                
            # Load JSON data
            with open(json_file, 'r') as f:
                models_data = json.load(f)
            
            # Save population
            save_population(population_uuid)
            
            # Save each model
            for model_id, model_data in models_data.items():
                save_model(population_uuid, model_id, model_data)
                
            logger.info(f"Imported population {population_uuid} with {len(models_data)} models")
        except Exception as e:
            logger.error(f"Error importing {json_file}: {e}")