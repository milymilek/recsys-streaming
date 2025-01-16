import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, date_format, from_unixtime

from recsys_lakehouse.lakehouse import layers, raw_data_source
from recsys_lakehouse.lakehouse.bronze.bronze import bronze_table_mapping
from recsys_lakehouse.lakehouse.silver import silver
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper


def load_table(spark: SparkSession, table_path: Path, table_name: str) -> DataFrame:
    return spark.read.parquet(str(table_path / table_name))


def main(spark: SparkSession):
    df = load_table(spark, Path(".datalake/silver/amazon_books_sample10000"), "books_reviews")
    df.show()
    df.printSchema()


if __name__ == "__main__":
    with spark_builder("test", "ERROR") as spark:
        main(spark)
