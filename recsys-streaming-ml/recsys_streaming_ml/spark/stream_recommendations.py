from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType

from recsys_streaming_ml.config import KAFKA_BROKER_URL, USER_ACTIONS_TOPIC

import json
from kafka import KafkaConsumer

KAFKA_SERVER_URI = 'localhost:9092'
TOPIC = 'recommendations'

def consume():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_SERVER_URI],
        auto_offset_reset='earliest',
    )
    
    for message in consumer:
        data = json.loads(message.value)
        print(f"Received: {data}")


def main():
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
        .option("startingOffsets", "earliest") \
        .load()

    values_df = df.selectExpr("CAST(value AS STRING) as json_data") \
                  .select(from_json(col("json_data"), schema).alias("data")) \
                  .select("data.*")

    query = values_df \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .trigger(processingTime='30 seconds') \
        .start()

    query.awaitTermination(30)
    query.stop()

    spark.stop()


if __name__ == "__main__":
    #consume()
    main()