from superset.db_engine_specs.sqlite import SqliteEngineSpec

SECRET_KEY = "key"

SQLALCHEMY_EXAMPLES = True  # Enable loading example DBs
DATABASES_ALLOWED_CLASS_BY_DRIVER = {
    "sqlite": SqliteEngineSpec,
}

# SQLALCHEMY_DATABASE_URI = "postgresql://airflow:airflow@postgres:5432/airflow"
