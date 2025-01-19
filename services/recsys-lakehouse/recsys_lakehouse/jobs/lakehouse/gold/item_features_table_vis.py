import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg2
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, rand, sum, when
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, TimestampType

from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.operator import TableOperator
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper, psql_conn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LayerConfig:
    app_name: str
    error_log_level: str
    dataset_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualisation Configuration")
    parser.add_argument("--app_name", type=str, default="Reviews table visualisation data aggregations.")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()

    return args


def item_features_mean(item_features_table: DataFrame, cursor, col):
    def item_features_mean_agg(item_features_table: DataFrame, col: str) -> DataFrame:
        return item_features_table.groupBy("main_category").agg(F.mean(col).alias(f"mean_{col}")).dropna().sort(f"mean_{col}")

    item_features_table_mean = item_features_mean_agg(item_features_table, col)
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS item_features_table_mean_{col} (
        main_category VARCHAR PRIMARY KEY,
        mean_{col} FLOAT
    )
    """
    cursor.execute(create_table_query)

    clear_table_query = f"TRUNCATE TABLE item_features_table_mean_{col}"
    cursor.execute(clear_table_query)

    insert_query = f"INSERT INTO item_features_table_mean_{col} (main_category, mean_{col}) VALUES (%s, %s)"
    data = item_features_table_mean.collect()
    cursor.executemany(insert_query, data)


@log_wrapper(enter="Reviews table visualisation data aggregations.", exit="Reviews table visualisation data aggregations created.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    gold_layer = layers.Gold(dataset_name=config.dataset_name)
    operator = TableOperator(spark)
    item_features_table = operator.read_table(gold_layer, gold_layer.tables["item_features"])

    with psql_conn() as cursor:
        item_features_mean(item_features_table, cursor, col="average_rating")
        item_features_mean(item_features_table, cursor, col="price")


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
