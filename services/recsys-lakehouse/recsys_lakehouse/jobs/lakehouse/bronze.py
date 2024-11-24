import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, from_unixtime
from recsys_lakehouse.lakehouse.bronze import TableFactory
from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.raw_data_source import JSONDataSource, RawDataSource, StreamDataSource
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper


@dataclass
class LayerConfig:
    app_name: str
    error_log_level: str
    _raw_data_source: str
    raw_data_path: str

    @property
    def raw_data_source(self) -> type[JSONDataSource] | type[StreamDataSource]:
        return JSONDataSource if self.raw_data_source == "JSONDataSource" else StreamDataSource  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bronze Layer Ingestion Configuration")
    parser.add_argument("--app_name", type=str, default="Bronze Layer Ingestion")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--raw_data_source", type=str, default="JSONDataSource")
    parser.add_argument("--raw_data_path", type=str, required=False, default="raw/amazon_books")
    args = parser.parse_args()

    return args

args = parse_args()
config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        _raw_data_source=args.raw_data_sources,
        raw_data_path=args.raw_data_path,
    )


@log_wrapper(enter="Loading JSON data into Spark DataFrame...", exit="Data loaded into Spark DataFrame.")
def load_json_to_spark(spark: SparkSession, file_path: Path):
    """Load JSON data into a Spark DataFrame."""

    df = spark.read.json(str(file_path))
    df.printSchema()
    return df


@log_wrapper(enter="Creating partition column date...", exit="Partition column date created.")
def create_partition_column(df):
    """Create a new column for partitioning the data."""

    df = df.withColumn("date", date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))
    return df


@log_wrapper(enter="Saving partitioned data...", exit="Partitioned data saved.")
def save_partitioned_data(df, partition_column: str, output_dir: Path):
    """Save Spark DataFrame as partitioned Parquet files."""

    df.write.mode("overwrite").partitionBy(partition_column).parquet(str(output_dir))


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession):
    raw_data_source = config.raw_data_source(spark)
    raw_layer = layers.Raw(source=raw_data_source, path=Path("amazon_books_sample10000"))
    bronze_layer = layers.Bronze(path=Path("amazon_books_sample10000"))
    # operator = LayerOperator(read_layer=Layers.RAW, write_layer=Layers.BRONZE)

    # files = operator.read_files()

    print(files)

    for filename in config.raw_data_files:
        raw_data_file_path = operator.
        raw_data_source = config.raw_data_source(spark, file)
        logging.info(f"Reading file {file}...")

        df = load_json_to_spark(spark, file)

        table = TableFactory.get_table_object(df, file.stem)
        table.partition_by(output_dir=operator.write_path(file.stem))


if __name__ == "__main__":
    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark)
