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

# df = spark.read.format("delta").load(".datalake/bronze/delta_table")
# df.show()

stream2 = spark.readStream.format("delta").load(".datalake/bronze/delta_table").writeStream.format("console").start()
stream2.awaitTermination()
