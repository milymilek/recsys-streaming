import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, rand, sum, when
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, TimestampType

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
    parser = argparse.ArgumentParser(description="Recommender Training")
    parser.add_argument("--app_name", type=str, default="Recommender Training")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books_sample10000")
    args = parser.parse_args()

    return args


def apply_mapping(df: DataFrame, **kwargs) -> DataFrame:
    for col, mapping_df in kwargs.items():
        mapping_col = next(iter(mapping_df.drop(col).columns), None)
        if mapping_col is None:
            raise ValueError(f"Column {col} not in mapping DataFrame")

        df = df.join(mapping_df, on=col, how="left")
        df = df.drop(col)
        df = df.withColumnRenamed(mapping_col, col)
    return df


def build_mapping_df(df: DataFrame, col: str, mapping_suffix: str = "_index") -> DataFrame:
    if col not in df.columns:
        raise ValueError(f"Column {col} not in DataFrame")

    mapping_rdd = df.select(col).rdd.map(lambda row: row[col]).distinct().zipWithIndex()
    mapping_df = mapping_rdd.toDF([col, col + mapping_suffix])
    return mapping_df


@log_wrapper(enter="Starting training recommender.", exit="Recommender training completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    operator = TableOperator(spark)
    gold = layers.Gold(dataset_name=config.dataset_name)
    reviews_df = operator.read_table(gold, table=gold.tables["reviews"])

    user_mapping = build_mapping_df(reviews_df, col="user_id")
    item_mapping = build_mapping_df(reviews_df, col="parent_asin")

    reviews_df_mapped = apply_mapping(reviews_df, user_id=user_mapping, parent_asin=item_mapping)

    (train_df, valid_df, test_df) = reviews_df_mapped.randomSplit([0.7, 0.2, 0.1])
    als = ALS(maxIter=10, rank=16, regParam=0.01, userCol="user_id", itemCol="parent_asin", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_df)
    train_df_preds = model.transform(train_df)
    valid_df_preds = model.transform(valid_df)

    evaluator = RegressionEvaluator(metricName="rmse", predictionCol="prediction", labelCol="rating")
    train_rmse = evaluator.evaluate(train_df_preds)
    valid_rmse = evaluator.evaluate(valid_df_preds)
    logging.info("Train RMSE: %s \nValid RMSE: %s", train_rmse, valid_rmse)


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        raw_data_source=args.raw_data_source,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
