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


def read_process_write_table(spark: SparkSession, config: LayerConfig, logger, silver_table_names, gold_table_name):
    silver_layer = layers.Silver(dataset_name=config.dataset_name)
    gold_layer = layers.Gold(dataset_name=config.dataset_name)
    operator = TableOperator(spark)

    silver_layer_dfs = {}
    for table_name in silver_table_names:
        logger.info("Reading table `%s` from silver layer.", table_name)
        silver_table = silver_layer.tables[table_name]
        silver_layer_dfs[table_name] = operator.read_table(silver_layer, silver_table)
        silver_layer_dfs[table_name].show()

    gold_table = gold_layer.tables[gold_table_name]
    df_table = gold_table.process(silver_layer_dfs)

    logger.info("Writing table `%s` to gold layer.", gold_table_name)
    operator.write_table(df_table, gold_table, gold_layer)
