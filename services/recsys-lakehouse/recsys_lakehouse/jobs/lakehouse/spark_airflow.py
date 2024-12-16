import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.operator import TableOperator
from recsys_lakehouse.lakehouse.raw_data_source import JSONDataSource, StreamDataSource
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books_sample10000")
    args = parser.parse_args()

    return args


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
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

    raw_layer = layers.Raw(source=raw_data_source, dataset_name=config.dataset_name)
    bronze_layer = layers.Bronze(dataset_name=config.dataset_name)
    operator = TableOperator(spark)

    for table_name, df in raw_layer.read_source().items():
        logger.info("Reading `%s`...", table_name)

        table = bronze_layer.tables[table_name]
        df_table = table.process(df)
        operator.write_table(df_table, table, bronze_layer)


if __name__ == "__main__":
    args = parse_args()
    print(args)
