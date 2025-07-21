import os
import json
import psycopg2
from psycopg2.extras import Json, DictCursor, register_default_jsonb
import uuid
import re
from utils import logger
from dotenv import load_dotenv

load_dotenv()

# Register JSONB adapter for proper JSON handling
register_default_jsonb(loads=json.loads)

# Environment variables for PostgreSQL connection
PG_HOST = os.environ.get("POSTGRES_HOST", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
PG_DB = os.environ.get("POSTGRES_DB", "neuroevolution")


def validate_input(input_value, input_type="string", max_length=None):
    """
    Validate and sanitize input to prevent SQL injection and other security issues.
    
    :param input_value: The input value to validate
    :type input_value: Any
    :param input_type: The expected type of the input (string, int, float, uuid)
    :type input_type: str
    :param max_length: Maximum allowed length for string inputs
    :type max_length: int
    :return: Sanitized input value or None if validation fails
    :rtype: Any
    """
    if input_value is None:
        return None
        
    try:
        # Type validation and conversion
        if input_type == "string":
            if not isinstance(input_value, str):
                logger.warning(f"Input validation failed: Expected string, got {type(input_value)}")
                return None
                
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[\\\'";]', '', input_value)
            
            # Check length if specified
            if max_length and len(sanitized) > max_length:
                logger.warning(f"Input validation failed: String exceeds maximum length of {max_length}")
                return None
                
            return sanitized
            
        elif input_type == "int":
            return int(input_value)
            
        elif input_type == "float":
            return float(input_value)
            
        elif input_type == "uuid":
            # Validate UUID format
            if not isinstance(input_value, str) or not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', input_value, re.I):
                logger.warning(f"Input validation failed: Invalid UUID format: {input_value}")
                return None
            return input_value
            
        else:
            logger.warning(f"Input validation failed: Unknown input type: {input_type}")
            return None
            
    except (ValueError, TypeError) as e:
        logger.warning(f"Input validation failed: {e}")
        return None


def get_db_connection():
    """
    Create and return a connection to the PostgreSQL database with enhanced security settings.
    
    :return: PostgreSQL database connection
    :rtype: psycopg2.extensions.connection
    """
    try:
        # Validate connection parameters to prevent injection
        host = validate_input(PG_HOST, "string", 255)
        port = validate_input(PG_PORT, "string", 10)
        user = validate_input(PG_USER, "string", 100)
        database = validate_input(PG_DB, "string", 100)
        
        # Don't validate password as it might contain special characters
        # that would be removed by our validation function
        
        if not all([host, port, user, database]):
            raise ValueError("Invalid database connection parameters")
            
        # Create connection with security settings
        conn = psycopg2.connect(
            host="localhost",
            port=port,
            user=user,
            password=PG_PASSWORD,
            database=database,
            # Set additional security options
            sslmode='prefer',  # Use SSL if available
            connect_timeout=10,  # Timeout to prevent hanging connections
            application_name='neuroevolution_broker'  # For audit identification
        )
        
        # Set session parameters for security
        with conn.cursor() as cursor:
            # Disable dangerous features
            cursor.execute("SET SESSION statement_timeout = '30s'")
            # Prevent SQL injection via search_path manipulation
            cursor.execute("SET SESSION search_path = '$user', public")
        
        return conn
    except Exception as e:
        # Log error but don't expose details that might help attackers
        logger.error(f"Database connection error: {type(e).__name__}")
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
            generation INTEGER DEFAULT 0,
            max_generations INTEGER DEFAULT 10,
            fitness_threshold FLOAT DEFAULT 0.95,
            fitness_history JSONB DEFAULT '[]',
            best_overall_fitness FLOAT DEFAULT 0.0,
            best_overall_uuid VARCHAR(36),
            original_params JSONB DEFAULT '{}',
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
    # Validate input to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return False
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO populations (uuid) VALUES (%s) ON CONFLICT (uuid) DO NOTHING",
            (validated_uuid,)
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving population {validated_uuid}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def save_population_with_metadata(population_uuid, generation=0, max_generations=10, fitness_threshold=0.95, 
                                 fitness_history=None, best_overall_fitness=0.0, best_overall_uuid=None, 
                                 original_params=None):
    """
    Save a population with genetic algorithm metadata to the database.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :param generation: Current generation number
    :type generation: int
    :param max_generations: Maximum number of generations
    :type max_generations: int
    :param fitness_threshold: Fitness threshold for convergence
    :type fitness_threshold: float
    :param fitness_history: History of fitness values
    :type fitness_history: list
    :param best_overall_fitness: Best fitness achieved so far
    :type best_overall_fitness: float
    :param best_overall_uuid: UUID of the population with best fitness
    :type best_overall_uuid: str
    :param original_params: Original parameters passed to the algorithm
    :type original_params: dict
    :return: True if successful, False otherwise
    :rtype: bool
    """
    # Validate input to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return False
    
    if fitness_history is None:
        fitness_history = []
    if original_params is None:
        original_params = {}
    if best_overall_uuid is None:
        best_overall_uuid = population_uuid
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO populations (uuid, generation, max_generations, fitness_threshold,
                                   fitness_history, best_overall_fitness, best_overall_uuid, original_params) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
            ON CONFLICT (uuid) DO UPDATE SET
                generation = EXCLUDED.generation,
                max_generations = EXCLUDED.max_generations,
                fitness_threshold = EXCLUDED.fitness_threshold,
                fitness_history = EXCLUDED.fitness_history,
                best_overall_fitness = EXCLUDED.best_overall_fitness,
                best_overall_uuid = EXCLUDED.best_overall_uuid,
                original_params = EXCLUDED.original_params
        """, (validated_uuid, generation, max_generations, fitness_threshold, 
              Json(fitness_history), best_overall_fitness, best_overall_uuid, Json(original_params)))
        
        conn.commit()
        logger.info(f"Successfully saved population {validated_uuid} with metadata")
        return True
    except Exception as e:
        logger.error(f"Error saving population {population_uuid} with metadata: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def get_population_metadata(population_uuid):
    """
    Get genetic algorithm metadata for a population.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :return: Dictionary with metadata or None if not found
    :rtype: dict or None
    """
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return None
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
            SELECT uuid, generation, max_generations, fitness_threshold, fitness_history,
                   best_overall_fitness, best_overall_uuid, original_params
            FROM populations 
            WHERE uuid = %s
        """, (validated_uuid,))
        
        row = cursor.fetchone()
        if row:
            return {
                'uuid': row['uuid'],
                'generation': row['generation'],
                'max_generations': row['max_generations'],
                'fitness_threshold': row['fitness_threshold'],
                'fitness_history': row['fitness_history'] if row['fitness_history'] else [],
                'best_overall_fitness': row['best_overall_fitness'],
                'best_overall_uuid': row['best_overall_uuid'],
                'original_params': row['original_params'] if row['original_params'] else {}
            }
        return None
    except Exception as e:
        logger.error(f"Error getting population metadata {population_uuid}: {e}")
        return None
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
    # Validate inputs to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return False
        
    validated_model_id = validate_input(model_id, input_type="string", max_length=50)
    if validated_model_id is None:
        logger.error(f"Invalid model ID: {model_id}")
        return False
    
    # Validate score if provided
    validated_score = None
    if score is not None:
        try:
            validated_score = float(score)
        except (ValueError, TypeError):
            logger.error(f"Invalid score value: {score}")
            return False
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure population exists
        save_population(validated_uuid)
        
        # If score is provided, include it in the model data
        if validated_score is not None:
            model_data["score"] = validated_score
        
        cursor.execute("""
        INSERT INTO models (population_uuid, model_id, data, score)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (population_uuid, model_id) DO UPDATE
        SET data = %s, score = %s
        """, (validated_uuid, validated_model_id, Json(model_data), 
              model_data.get("score", 0), Json(model_data), model_data.get("score", 0)))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving model {validated_model_id} for population {validated_uuid}: {e}")
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
    # Validate inputs to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return False
        
    validated_model_id = validate_input(model_id, input_type="string", max_length=50)
    if validated_model_id is None:
        logger.error(f"Invalid model ID: {model_id}")
        return False
    
    # Validate score
    try:
        validated_score = float(score)
    except (ValueError, TypeError):
        logger.error(f"Invalid score value: {score}")
        return False
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the score in both the score column and the JSONB data
        cursor.execute("""
        UPDATE models
        SET score = %s, data = jsonb_set(data, '{score}', %s::jsonb)
        WHERE population_uuid = %s AND model_id = %s
        """, (validated_score, json.dumps(validated_score), validated_uuid, validated_model_id))
        
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating score for model {validated_model_id} in population {validated_uuid}: {e}")
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
    # Validate input to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return {}
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()  # Use regular cursor instead of DictCursor
        
        cursor.execute("""
        SELECT model_id, data, score
        FROM models
        WHERE population_uuid = %s
        """, (validated_uuid,))
        
        rows = cursor.fetchall()
        logger.debug(f"Found {len(rows)} models for population {validated_uuid}")
        
        models = {}
        for row in rows:
            model_id = row[0]
            data = row[1]
            score = row[2]
            data['score'] = score if score is not None else 0
            
            # Ensure data is a dictionary
            if isinstance(data, dict):
                models[model_id] = data
            elif isinstance(data, str):
                # If it's a string, try to parse it as JSON
                try:
                    models[model_id] = json.loads(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON data for model {model_id}: {e}")
                    continue
            else:
                logger.error(f"Unexpected data type {type(data)} for model {model_id}")
                continue
        
        logger.debug(f"Successfully processed {len(models)} models for population {validated_uuid}")
        return models
    except Exception as e:
        logger.error(f"Error retrieving population {validated_uuid}: {e}")
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
    # Validate input to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return False
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT EXISTS(SELECT 1 FROM populations WHERE uuid = %s)
        """, (validated_uuid,))
        
        exists = cursor.fetchone()[0]
        logger.debug(f"Population {validated_uuid} exists: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Error checking if population {validated_uuid} exists: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_best_fitness_from_population(population_uuid):
    """
    Get the best fitness score from a specific population.
    
    :param population_uuid: UUID of the population
    :type population_uuid: str
    :return: Best fitness score, or 0.0 if no models or error
    :rtype: float
    """
    # Validate input to prevent SQL injection
    validated_uuid = validate_input(population_uuid, input_type="uuid")
    if validated_uuid is None:
        logger.error(f"Invalid population UUID format: {population_uuid}")
        return 0.0
        
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the maximum score from the population
        cursor.execute("""
        SELECT MAX(score) 
        FROM models 
        WHERE population_uuid = %s AND score IS NOT NULL
        """, (validated_uuid,))
        
        result = cursor.fetchone()
        
        if result and result[0] is not None:
            best_fitness = float(result[0])
            logger.debug(f"Best fitness for population {validated_uuid}: {best_fitness}")
            return best_fitness
        else:
            logger.warning(f"No scores found for population {validated_uuid}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error getting best fitness for population {validated_uuid}: {e}")
        return 0.0
    finally:
        if conn:
            conn.close()


