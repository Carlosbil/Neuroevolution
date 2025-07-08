import os
import json
import psycopg2
from psycopg2.extras import Json, DictCursor
import uuid
import re
from utils import logger
from dotenv import load_dotenv

load_dotenv()

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
            host=host,
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
        cursor = conn.cursor(DictCursor)
        
        cursor.execute("""
        SELECT model_id, data, score
        FROM models
        WHERE population_uuid = %s
        """, (validated_uuid,))
        
        models = {}
        for row in cursor.fetchall():
            models[row['model_id']] = row['data']
        
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
        SELECT EXISTS(SELECT 1 FROM models WHERE population_uuid = %s)
        """, (validated_uuid,))
        
        return cursor.fetchone()[0]
    except Exception as e:
        logger.error(f"Error checking if population {validated_uuid} exists: {e}")
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
            # Validate file path to prevent path traversal attacks
            if not os.path.abspath(json_file).startswith(os.path.abspath(models_path)):
                logger.error(f"Security warning: Attempted access to file outside models directory: {json_file}")
                continue
                
            # Extract UUID from filename
            filename = os.path.basename(json_file)
            population_uuid = os.path.splitext(filename)[0]
            
            # Validate UUID format
            validated_uuid = validate_input(population_uuid, input_type="uuid")
            if validated_uuid is None:
                logger.warning(f"Skipping file with invalid UUID format: {filename}")
                continue
            
            # Skip files with _best50percent suffix
            if "_best50percent" in validated_uuid:
                continue
                
            # Load JSON data with size limit to prevent memory attacks
            file_size = os.path.getsize(json_file)
            max_size = 50 * 1024 * 1024  # 50MB limit
            if file_size > max_size:
                logger.warning(f"Skipping oversized file {json_file} ({file_size} bytes)")
                continue
                
            with open(json_file, 'r') as f:
                models_data = json.load(f)
            
            # Validate data structure
            if not isinstance(models_data, dict):
                logger.warning(f"Invalid data structure in {json_file}, expected dictionary")
                continue
            
            # Save population
            save_population(validated_uuid)
            
            # Save each model with validation
            for model_id, model_data in models_data.items():
                # Validate model_id
                validated_model_id = validate_input(model_id, input_type="string", max_length=50)
                if validated_model_id is None:
                    logger.warning(f"Skipping model with invalid ID: {model_id}")
                    continue
                    
                # Validate model_data is a dictionary
                if not isinstance(model_data, dict):
                    logger.warning(f"Skipping model with invalid data structure: {model_id}")
                    continue
                    
                save_model(validated_uuid, validated_model_id, model_data)
                
            logger.info(f"Imported population {validated_uuid} with {len(models_data)} models")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {json_file}: {e}")
        except Exception as e:
            logger.error(f"Error importing {json_file}: {e}")