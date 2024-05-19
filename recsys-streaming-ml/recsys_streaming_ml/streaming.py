from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, struct, array
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, Union, Dict


KAFKA_BROKER_URL = "kafka0:9093"
RECOMMENDATIONS_TOPIC = "recommendations"
USER_ACTIONS_TOPIC = "users.actions"


def predict_batch_fn():
    # load model from checkpoint
    import torch    
    device = torch.device("cuda")
    model = Net().to(device)
    checkpoint = load_checkpoint(checkpoint_dir)
    model.load_state_dict(checkpoint['model'])

    # define predict function in terms of numpy arrays
    def predict(inputs: np.ndarray) -> np.ndarray:
        torch_inputs = torch.from_numpy(inputs).to(device)
        outputs = model(torch_inputs)
        return outputs.cpu().detach().numpy()
    
    return predict


def main():
    spark = SparkSession.builder \
        .appName("KafkaRead") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()

    schema = StructType([
        StructField("user_id", StringType(), True)
    ])

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER_URL) \
        .option("subscribe", RECOMMENDATIONS_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df.selectExpr("CAST(value AS STRING) as json_value") \
        .select(from_json(col("json_value"), schema).alias("data")) \
        .select("data.*")

    # create standard PandasUDF from predict function
    make_recommendations = predict_batch_udf(predict_batch_fn,
                            input_tensor_shapes=[[1,28,28]],
                            return_type=ArrayType(FloatType()),
                            batch_size=1000)

    df = spark.read.parquet("/path/to/test/data")
    preds = df.withColumn("preds", mnist('data')).collect()

    query = df_parsed.writeStream \
        .outputMode("append") \
        .format("console") \
        #.trigger(processingTime='15 seconds') \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()