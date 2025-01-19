import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from recsys_lakehouse.lakehouse.layers import Raw
from recsys_lakehouse.lakehouse.raw_data_source import JSONDataSource
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class LayerConfig:
    app_name: str
    error_log_level: str
    dataset_name: str
    n_last_months: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampling Config")
    parser.add_argument("--app_name", type=str, default="Raw Data Sampling")
    parser.add_argument("--error_log_level", type=str, default="INFO")
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books")
    parser.add_argument("--n_last_months", type=int, required=False, default=3)

    args = parser.parse_args()

    return args


@log_wrapper(enter="Starting raw data sampling.", exit="Raw data sampling completed successfully.")
def main(spark: SparkSession, config) -> None:
    data_source = JSONDataSource(
        base_path=Path(f".datalake/raw/{config.dataset_name}"),
        tables={"books_reviews": "Books.jsonl", "books_metadata": "meta_Books.jsonl"},
        spark=spark,
    )

    print("preparing...")

    raw_layer = Raw(source=data_source, dataset_name=config.dataset_name)
    books_reviews_raw: DataFrame = raw_layer.read_source("books_reviews")  # type: ignore
    books_metadata_raw: DataFrame = raw_layer.read_source("books_metadata")  # type: ignore

    print("Books read")

    books_reviews = books_reviews_raw.withColumn("timestampMonthYear", F.from_unixtime(F.col("timestamp") / 1000, format="yyyy-MM"))
    max_timestamp = books_reviews.agg(F.max("timestampMonthYear").alias("timestampMonthYear"))
    ts = (
        max_timestamp.withColumn("timestampMonthYearSub", F.date_format(F.add_months(F.col("timestampMonthYear"), -3), "yyyy-MM"))
        .select("timestampMonthYearSub")
        .first()[0]  # type: ignore
    )
    books_reviews_filtered = books_reviews.filter(F.col("timestampMonthYear") > ts)

    print("Books filtered")

    items_distinct = books_reviews_filtered.select("parent_asin").distinct()
    books_metadata_filtered = books_metadata_raw.join(items_distinct, on=["parent_asin"], how="right")

    print(f"Books reviews count: {books_reviews_raw.count()} \nBooks metadata count: {books_metadata_raw.count()}")
    print(
        f"Books reviews (filtered) count: {books_reviews_filtered.count()} \nBooks metadata (filtered) count: {books_metadata_filtered.count()}"
    )

    books_reviews_filtered.write.mode("overwrite").json(f".datalake/raw/amazon_books_from_{ts}/Books.jsonl")
    books_metadata_filtered.write.mode("overwrite").json(f".datalake/raw/amazon_books_from_{ts}/meta_Books.jsonl")


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
        n_last_months=args.n_last_months,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
