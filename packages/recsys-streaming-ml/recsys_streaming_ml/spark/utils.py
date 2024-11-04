from pyspark.sql import SparkSession

def _set_spark_envs():
    import os
    if not os.environ['DISABLE_SPARK_ENVS']:
        os.environ['PYSPARK_PYTHON'] = 'C:/Users/Milosz/AppData/Local/pypoetry/Cache/virtualenvs/recsys-streaming-ml-Mj1TWbkU-py3.10/Scripts/python.exe'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/Milosz/AppData/Local/pypoetry/Cache/virtualenvs/recsys-streaming-ml-Mj1TWbkU-py3.10/Scripts/python.exe'

def spark():
    """
    Create a SparkSession.
    """
    _set_spark_envs()

    spark = SparkSession.builder \
        .appName("CreateDataFrameFromDict") \
        .master("local[4]") \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    return spark


        # .config("spark.executor.memory", "4g") \

def spark_structured_streaming():
    """
    Create a SparkSession with Structured Streaming support.
    """
    _set_spark_envs()

    spark = SparkSession.builder \
        .appName("KafkaRead") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()
    return spark
        

