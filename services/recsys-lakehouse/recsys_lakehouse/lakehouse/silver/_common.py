import argparse
from dataclasses import dataclass

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.operator import TableOperator


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


def clean_bronze_write_silver(spark: SparkSession, config: LayerConfig, logger, table_name) -> None:
    bronze_layer = layers.Bronze(dataset_name=config.dataset_name)
    silver_layer = layers.Silver(dataset_name=config.dataset_name)
    operator = TableOperator(spark)

    logger.info("Reading table `%s` from bronze layer.", table_name)
    bronze_table = bronze_layer.tables[table_name]
    df = operator.read_table(bronze_layer, bronze_table)

    df.show()

    silver_table = silver_layer.tables[table_name]
    df_table = silver_table.process(df)

    logger.info("Writing table `%s` to silver layer.", table_name)
    operator.write_table(df_table, silver_table, silver_layer)
