from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType

from recsys_streaming_ml.spark.utils import spark_structured_streaming
from recsys_streaming_ml.config import KAFKA_BROKER_URL, USER_ACTIONS_TOPIC, TRAINING_OFFSET
from recsys_streaming_ml.data.process_data import process_data
from recsys_streaming_ml.model.train import train


def full_retraining_process(users_actions_df, batch_id):
    if not users_actions_df.isEmpty():
        print(f"Showing data for batch: {batch_id}")
        users_actions_df.show()

        process_data(df_rating_stream=users_actions_df.toPandas())
        train(epochs=3, batch_size=16, validation_frac=0.2, cuda=False, seed=42)


def stream():
    session: SparkSession = spark_structured_streaming()

    schema = StructType([
        StructField("parent_asin", StringType(), True),
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
        .foreachBatch(full_retraining_process) \
        .trigger(processingTime=TRAINING_OFFSET) \
        .start()

    query.awaitTermination()


def run():
    print("SCRIPT: Stream users actions - START")

    stream()

    print("SCRIPT: Train model - END")


if __name__ == "__main__":
    run()