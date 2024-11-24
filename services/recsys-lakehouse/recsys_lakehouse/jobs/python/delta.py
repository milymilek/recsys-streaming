import random

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("DeltaExample")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

letter = ["a", "b", "c", "d", "e"]
data = [(random.choice(letter), random.randint(1, 5)) for _ in range(3)]
columns = ["Letter", "Number"]
df = spark.createDataFrame(data, columns)
df.show()

df.write.format("delta").mode("overwrite").save(".datalake/bronze/delta_table")

delta_table = spark.read.format("delta").load(".datalake/bronze/delta_table")
delta_table.groupBy("Letter").count().show()
