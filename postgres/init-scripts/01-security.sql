-- Script de inicialización para configurar seguridad en PostgreSQL

-- Revocar todos los permisos públicos por defecto
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- Crear un rol específico para la aplicación con permisos limitados
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_role') THEN
      CREATE ROLE app_role;
   END IF;
END
$$;

-- Asignar permisos específicos al rol de la aplicación
GRANT USAGE ON SCHEMA public TO app_role;

-- Asignar el rol al usuario de la aplicación
GRANT app_role TO CURRENT_USER;

-- Configurar parámetros de seguridad a nivel de base de datos
ALTER DATABASE CURRENT_DATABASE() SET search_path = "$user", public;

-- Crear función para prevenir inyección SQL en consultas dinámicas
CREATE OR REPLACE FUNCTION prevent_sql_injection(input_text TEXT) 
RETURNS TEXT AS $$
BEGIN
    -- Eliminar caracteres potencialmente peligrosos
    RETURN regexp_replace(input_text, E'[\\\'\";\\\\]', '', 'g');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Crear extensión pgcrypto para funciones de hash seguras
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Configurar auditoría básica para operaciones críticas
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    action_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_name TEXT NOT NULL,
    action TEXT NOT NULL,
    object_affected TEXT,
    query TEXT
);

-- Función para registrar acciones en el log de auditoría
CREATE OR REPLACE FUNCTION log_action()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log(user_name, action, object_affected, query)
    VALUES (CURRENT_USER, TG_OP, TG_TABLE_NAME, current_query());
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Aplicar trigger de auditoría a las tablas principales
CREATE TRIGGER populations_audit
AFTER INSERT OR UPDATE OR DELETE ON populations
FOR EACH STATEMENT EXECUTE FUNCTION log_action();

CREATE TRIGGER models_audit
AFTER INSERT OR UPDATE OR DELETE ON models
FOR EACH STATEMENT EXECUTE FUNCTION log_action();

-- Configurar límites de conexión por usuario
ALTER ROLE app_role CONNECTION LIMIT 20;

-- Configurar timeout de sesión para prevenir sesiones abandonadas
ALTER ROLE app_role SET statement_timeout = '300s';

-- Mensaje de finalización
DO $$
BEGIN
    RAISE NOTICE 'Configuración de seguridad aplicada correctamente';
END
$$;