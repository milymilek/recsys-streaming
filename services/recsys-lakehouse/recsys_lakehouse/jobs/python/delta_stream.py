import random
import time

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("DeltaExample")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# df = spark.read.format("delta").option("versionAsOf", 2).load(".datalake/bronze/delta_table").show()


# streaming_df = spark.readStream.format("delta").load(".datalake/bronze/delta_table")
streaming_df = spark.readStream.format("rate").load()
stream = (
    streaming_df.selectExpr("value as Number")
    .writeStream.format("delta")
    .option("checkpointLocation", ".datalake/bronze/delta_table_checkpoints")
    .start(".datalake/bronze/delta_table")
)
stream.awaitTermination()
# query = streaming_df.writeStream.format("console").outputMode("append").start()
# query.awaitTermination()
