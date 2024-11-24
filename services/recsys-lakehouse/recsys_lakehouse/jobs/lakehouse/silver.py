import logging
import os
from abc import abstractmethod
from enum import Enum
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, from_unixtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Layers(str, Enum):
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class LayerOperator:
    def __init__(self, read_layer: Layers, write_layer: Layers, raw="raw"):
        self._raw = raw
        self._read_layer = read_layer
        self._write_layer = write_layer

    @property
    def base_path(self) -> Path:
        return Path(".datalake")

    def read_files(self) -> list[Path]:
        return list((self.base_path / self._raw).iterdir())

    def read_path(self, file_name: str) -> Path:
        return self.base_path / self._read_layer.value / f"amazon_books/data_source=http_github/{file_name}"

    def write_path(self, file_name: str) -> Path:
        return self.base_path / self._write_layer.value / f"amazon_books/data_source=http_github/{file_name}"


class Table:
    def __init__(self, df):
        self._df = df

    @abstractmethod
    def partition_by(self, output_dir: Path):
        pass


class BooksTable(Table):
    partition_col = "date"

    def partition_by(self, output_dir: Path):
        self._df = self._df.withColumn(self.partition_col, date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir))


class MetaBooksTable(Table):
    partition_col = "main_category"

    def partition_by(self, output_dir: Path):
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir))


class TableFactory:
    @staticmethod
    def get_table_object(df, table_name: str) -> Table:
        table_classes = {
            "Books": BooksTable,
            "meta_Books": MetaBooksTable,
        }
        c = table_classes.get(table_name)

        if c is None:
            raise ValueError(f"Table {table_name} not found.")

        return c(df)


spark = SparkSession.builder.appName("Bronze Layer Ingestion").config("spark.sql.parquet.compression.codec", "snappy").getOrCreate()  # type: ignore


def load_json_to_spark(file_path: Path):
    """
    Load JSON data into a Spark DataFrame.
    """
    logging.info(f"Loading JSON data from {str(file_path)} into Spark DataFrame...")

    df = spark.read.json(str(file_path))
    df.printSchema()

    logging.info("Data loaded into Spark DataFrame.")
    return df


def create_partition_column(df):
    """
    Create a new column for partitioning the data.
    """
    logging.info(f"Creating partition column date...")

    df = df.withColumn("date", date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))

    logging.info(f"Partition column date created.")
    return df


def save_partitioned_data(df, partition_column: str, output_dir: Path):
    """
    Save Spark DataFrame as partitioned Parquet files.
    """
    logging.info(f"Saving data partitioned by {partition_column} to {output_dir}...")

    df.write.mode("overwrite").partitionBy(partition_column).parquet(str(output_dir))

    logging.info(f"Data saved to {output_dir} in partitioned format.")


def main():
    logging.info(f"\n\n\n {'='*5}Starting normalization.{'='*5}\n\n\n")

    kwargs = {}
    if os.getenv("RAW_PATH") is not None:
        kwargs = {"raw": os.environ["RAW_PATH"]}

    operator = LayerOperator(read_layer=Layers.RAW, write_layer=Layers.BRONZE, **kwargs)

    files = operator.read_files()

    for file in files:
        logging.info(f"Reading file {file}...")

        df = load_json_to_spark(file)

        table = TableFactory.get_table_object(df, file.stem)
        table.partition_by(output_dir=operator.write_path(file.stem))

    logging.info(f"\n\n\n {'='*5}Ingestion completed successfully.{'='*5}\n\n\n")


if __name__ == "__main__":
    main()
    spark.stop()
