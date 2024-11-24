import os
import random
import time
from contextlib import contextmanager

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import IntegerType, StructField, StructType

DEBUG = os.getenv("DEBUG", True)

APP_NAME = "Read Kafka Producer: users_actions"


@contextmanager
def spark_builder(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()  # type: ignore
    spark.sparkContext.setLogLevel("ERROR")

    yield spark
    spark.stop()


def build_users_actions_stream(spark: SparkSession):
    if DEBUG:
        host = "localhost"
        port = "9999"
        return spark.readStream.format("socket").option("host", host).option("port", port).load()
    raise RuntimeError("Kafka not handled yet.")


def main():
    with spark_builder(app_name=APP_NAME) as spark:
        users_actions_stream = build_users_actions_stream(spark)
        users_actions_schema = StructType([StructField("user_id", IntegerType(), True), StructField("item_id", IntegerType(), True)])


df_users_actions_stream_parsed = users_actions_stream.select(from_json(col("value"), users_actions_schema).alias("data"))
df_users_actions_stream = df_users_actions_stream_parsed.select("data.user_id", "data.item_id")


query = df_users_actions_stream.writeStream.outputMode("append").format("console").option("truncate", "false").start()
query.awaitTermination()


if __name__ == "__main__":
    main()
