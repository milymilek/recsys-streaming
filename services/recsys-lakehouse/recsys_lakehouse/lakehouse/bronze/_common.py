import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse.layers import Bronze, Raw
from recsys_lakehouse.lakehouse.operator import TableOperator
from recsys_lakehouse.lakehouse.raw_data_source import JSONDataSource, StreamDataSource
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper


@dataclass
class LayerConfig:
    app_name: str
    error_log_level: str
    raw_data_source: str
    dataset_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bronze Layer Configuration")
    parser.add_argument("--app_name", type=str, default="Bronze Layer - Ingestion")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--raw_data_source", type=str, default="JSONDataSource")
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books_from_2023-06")
    args = parser.parse_args()

    return args


def read_process_write_table(spark: SparkSession, config, table_name: str) -> None:
    if config.raw_data_source == "JSONDataSource":
        raw_data_source = JSONDataSource(
            base_path=Path(f".datalake/raw/{config.dataset_name}"),
            tables={"books_reviews": "Books.jsonl", "books_metadata": "meta_Books.jsonl"},
            spark=spark,
        )
    elif config.raw_data_source == "StreamDataSource":
        raw_data_source = StreamDataSource(tables={"books_reviews": ""}, spark=spark)
    else:
        raise ValueError(f"Unknown raw data source: {config.raw_data_source}")

    raw_layer = Raw(source=raw_data_source, dataset_name=config.dataset_name)
    books_reviews_raw = raw_layer.read_source(table_name)
    bronze_layer = Bronze(dataset_name=config.dataset_name)
    books_reviews_bronze_table = bronze_layer.tables[table_name]
    books_reviews_bronze = books_reviews_bronze_table.process(books_reviews_raw)
    books_reviews_bronze.show()
    operator = TableOperator(spark)
    operator.write_table(books_reviews_bronze, books_reviews_bronze_table, bronze_layer)
