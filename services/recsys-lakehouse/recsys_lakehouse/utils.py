import logging
from contextlib import contextmanager

import psycopg2

# TODO: move to airflow scheduler's envs
POSTGRES_HOST = "postgres"
POSTGRES_PORT = "5432"
POSTGRES_DB = "airflow"
POSTGRES_USER = "airflow"
POSTGRES_PASSWORD = "airflow"


def log_wrapper(enter: str = "Entering", exit: str = "Exiting", log_fn=print):
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_fn("[%s]: %s", func.__name__, enter)
            retval = func(*args, **kwargs)
            log_fn("[%s]: %s", func.__name__, exit)
            return retval

        return wrapper

    return decorator


@contextmanager
def psql_conn():
    conn = psycopg2.connect(dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
    conn.autocommit = True
    cursor = conn.cursor()

    yield cursor

    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()
