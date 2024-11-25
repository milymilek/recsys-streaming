import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, date_format, from_unixtime

from recsys_lakehouse.lakehouse import layers, raw_data_source, silver
from recsys_lakehouse.lakehouse.bronze import bronze_table_mapping
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper


@dataclass
class LayerConfig:
    app_name: str
    error_log_level: str
    dataset_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver Layer Configuration")
    parser.add_argument("--app_name", type=str, default="Silver Layer - Normalization")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books_sample10000")
    args = parser.parse_args()

    return args


args = parse_args()
config = LayerConfig(
    app_name=args.app_name,
    error_log_level=args.error_log_level,
    dataset_name=args.dataset_name,
)


@log_wrapper(enter="Loading table...", exit="Table loaded.")
def load_table(spark: SparkSession, table_path: Path, table_name: str) -> DataFrame:
    return spark.read.parquet(str(table_path / table_name))


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession):
    bronze_layer = layers.Bronze(path=Path(config.dataset_name))
    silver_layer = layers.Silver(path=Path(config.dataset_name), spark=spark)

    # for raw batch data
    for table_name in bronze_table_mapping.keys():
        print(table_name)

        df = load_table(spark, bronze_layer.path, table_name)
        df.show()

        table = {"books": silver.BooksReviewsTable, "meta_books": silver.BooksMetadataTable}[table_name](df)
        table.process()
        silver_layer.write_table(table._df, table)


if __name__ == "__main__":
    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark)
