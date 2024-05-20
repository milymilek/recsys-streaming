from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType

from recsys_streaming_ml.config import KAFKA_BROKER_URL, USER_ACTIONS_TOPIC, TRAINING_OFFSET
from recsys_streaming_ml.model.train import train_placeholer


def start_streaming():
    spark = SparkSession.builder \
        .appName("KafkaRead") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1") \
        .getOrCreate()

    schema = StructType([
        StructField("asin", StringType(), True),
        StructField("user_id", StringType(), True)
    ])

    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER_URL) \
        .option("subscribe", USER_ACTIONS_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    data = df.selectExpr("CAST(value AS STRING) as json_string")
    data = data.select(from_json("json_string", schema).alias("data")).select("data.*")

    query = data.writeStream \
        .foreachBatch(train_placeholer) \
        .trigger(processingTime=TRAINING_OFFSET) \
        .start()

    query.awaitTermination()