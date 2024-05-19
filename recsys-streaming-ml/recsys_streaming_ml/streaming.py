from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# spark_jars = ("{},{},{},{},{}".format("../jars/commons-pool2-2.12.0.jar",
#                                     "../jars/kafka-clients-3.7.0.jar",
#                                     "../jars/spark-sql-kafka-0-10_2.12-3.5.1.jar",
#                                     "../jars/spark-streaming-kafka-0-10_2.12-3.5.1.jar",
#                                     "../jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar"))

# # Initialize a SparkSession
# spark = SparkSession \
#     .builder \
#     .config("spark.jars", spark_jars) \
#     .appName("KafkaStructuredStreamingExample") \
#     .getOrCreate()

spark = SparkSession \
    .builder \
    .appName("Streaming from Kafka") \
    .config("spark.streaming.stopGracefullyOnShutdown", True) \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1') \
    .config("spark.sql.shuffle.partitions", 4) \
    .master("local[*]") \
    .getOrCreate()

# Define Kafka parameters
kafka_bootstrap_servers = "kafka:9092"
kafka_topic = "user_activity"

# Read messages from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()

# Select the key and value columns and cast them to strings
kafka_df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

# Print the schema of the incoming data
kafka_df.printSchema()

# Define the query that writes the streaming data to the console
query = kafka_df \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Wait for the termination signal from the user
query.awaitTermination()