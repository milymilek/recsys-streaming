import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.operator import TableOperator
from recsys_lakehouse.lakehouse.silver import silver_table_mapping
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


@log_wrapper(enter="Loading table...", exit="Table loaded.")
def load_table(spark: SparkSession, table_path: Path, table_name: str) -> DataFrame:
    return spark.read.parquet(str(table_path / table_name))


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    bronze_layer = layers.Bronze(dataset_name=config.dataset_name)
    silver_layer = layers.Silver(dataset_name=config.dataset_name)
    operator = TableOperator(spark)

    for table_name, bronze_table in bronze_layer.tables.items():
        logger.info("Reading table `%s` from bronze layer.", table_name)
        df = operator.read_table(bronze_layer, bronze_table)
        df.show()

        silver_table = silver_layer.tables[table_name]
        df_table = silver_table.process(df)

        logger.info("Writing table `%s` to silver layer.", table_name)
        operator.write_table(df_table, silver_table, silver_layer)


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
