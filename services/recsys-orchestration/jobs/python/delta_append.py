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


letter = ["a", "b", "c", "d", "e"]

while True:
    data = [(random.choice(letter), 0)]
    columns = ["Letter", "Number"]
    df = spark.createDataFrame(data, columns)
    df.write.format("delta").mode("append").save(".datalake/bronze/delta_table")
    print("appending data...")
    time.sleep(10)
