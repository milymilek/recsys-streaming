import logging
import os
from pathlib import Path

# import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration parameters (customize as needed)
DATA_SOURCE_URL = "https://example.com/data"  # Example URL for data source
RAW_DATA_DIR = Path(".datalake/bronze")  # Directory to store raw data
FILE_NAME = "meta_Books_sample.json"  # Example file name to download

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("Bronze Layer Ingestion").config("spark.sql.parquet.compression.codec", "snappy").getOrCreate()


# def download_data(url: str, save_path: Path):
#     """
#     Download data from an HTTP endpoint and save it to a local file.
#     """
#     logging.info(f"Downloading data from {url}...")
#     response = requests.get(url)
#     response.raise_for_status()  # Raise an error for failed requests

#     with open(save_path, "wb") as f:
#         f.write(response.content)
#     logging.info(f"Data downloaded and saved to {save_path}")


def load_json_to_spark(file_path: Path):
    """
    Load JSON data into a Spark DataFrame.
    """
    logging.info(f"Loading JSON data from {str(file_path)} into Spark DataFrame...")
    df = spark.read.json(str(file_path))
    logging.info("Data loaded into Spark DataFrame.")
    return df


def save_partitioned_data(df, partition_column: str, output_dir: Path):
    """
    Save Spark DataFrame as partitioned Parquet files.
    """
    logging.info(f"Saving data partitioned by {partition_column} to {output_dir}...")
    df.write.mode("overwrite").partitionBy(partition_column).parquet(str(output_dir))
    logging.info(f"Data saved to {output_dir} in partitioned format.")


def main():
    # Step 1: Download raw data
    json_path = RAW_DATA_DIR / FILE_NAME
    # download_data(DATA_SOURCE_URL, json_path)

    # with open(RAW_DATA_DIR / "file.json", "w") as f:
    #     f.write('{"user_id": 1, "product_id": 10, "rating": 5}\n')

    # Step 2: Load JSON data into a Spark DataFrame
    df = load_json_to_spark(json_path)

    # # Step 3: Perform any data filtering/cleanup (optional)
    # # Example: Filter records where `user_id` is not null
    df = df.filter(col("user_id").isNotNull())

    # # Step 4: Save DataFrame to Parquet format, partitioned by `user_id`
    save_partitioned_data(df, partition_column="user_id", output_dir=RAW_DATA_DIR / "partitioned")

    logging.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
    spark.stop()
