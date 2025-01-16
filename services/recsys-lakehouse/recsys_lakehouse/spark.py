from contextlib import contextmanager

from pyspark.sql import SparkSession


@contextmanager
def spark_builder(app_name: str, error_log_level: str = "ERROR"):
    spark = SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()  # type: ignore
    spark.sparkContext.setLogLevel(error_log_level)

    yield spark
    spark.stop()
