import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# Define PostgreSQL connection details
POSTGRES_HOST = "postgres"
POSTGRES_PORT = "5432"
POSTGRES_DB = "airflow"
POSTGRES_USER = "airflow"
POSTGRES_PASSWORD = "airflow"
TABLE_NAME = "row_counts"

# Initialize PySpark session
spark = SparkSession.builder.appName("Row Count to PostgreSQL").master("local[*]").getOrCreate()

# Step 1: Read the data from the datalake
data_path = ".datalake/gold/amazon_books_sample10000/reviews"
data_format = "parquet"  # Change to "csv" or "json" if needed
df = spark.read.format(data_format).load(data_path)

# Step 2: Count the number of rows
row_count = df.count()

df.show()

print("row_count", row_count)

# Step 3: Create PostgreSQL table if it doesn't exist
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255),
    row_count INT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

insert_query = f"""
INSERT INTO {TABLE_NAME} (dataset_name, row_count) 
VALUES (%s, %s);
"""

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
    conn.autocommit = True
    cursor = conn.cursor()

    # Execute table creation query
    cursor.execute(create_table_query)

    # Step 4: Insert the row count into the table
    cursor.execute(insert_query, ("amazon_books_sample10000_reviews", row_count))

    print(f"Row count ({row_count}) successfully saved to PostgreSQL table '{TABLE_NAME}'.")

except Exception as e:
    print(f"Error: {e}")

finally:
    if "cursor" in locals():
        cursor.close()
    if "conn" in locals():
        conn.close()

# Stop the Spark session
spark.stop()
