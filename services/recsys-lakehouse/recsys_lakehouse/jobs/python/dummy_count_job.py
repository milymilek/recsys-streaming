from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonCounter").getOrCreate()

# write spark script that sums an array of 1000 elements
data = [1] * 1_000_000
distData = spark.sparkContext.parallelize(data)
count = distData.count()
print(f"count: {count}")

spark.stop()
