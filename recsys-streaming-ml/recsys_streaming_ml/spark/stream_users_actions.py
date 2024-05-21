from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType

from recsys_streaming_ml.spark.utils import spark_structured_streaming
from recsys_streaming_ml.config import KAFKA_BROKER_URL, USER_ACTIONS_TOPIC, TRAINING_OFFSET
from recsys_streaming_ml.model.train import train_placeholer


def main():
    session: SparkSession = spark_structured_streaming()

    schema = StructType([
        StructField("asin", StringType(), True),
        StructField("user_id", StringType(), True)
    ])

    df = session \
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


if __name__ == "__main__":
    main()