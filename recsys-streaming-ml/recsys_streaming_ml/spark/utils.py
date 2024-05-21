from pyspark.sql import SparkSession

def _set_spark_envs():
    import os
    os.environ['PYSPARK_PYTHON'] = 'C:/Users/Milosz/AppData/Local/pypoetry/Cache/virtualenvs/recsys-streaming-ml-Mj1TWbkU-py3.10/Scripts/python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/Milosz/AppData/Local/pypoetry/Cache/virtualenvs/recsys-streaming-ml-Mj1TWbkU-py3.10/Scripts/python.exe'

def spark():
    """
    Create a SparkSession.
    """
    _set_spark_envs()

    spark = SparkSession.builder \
        .appName("CreateDataFrameFromDict") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", 10) \
        .getOrCreate()
    return spark


        # .config("spark.executor.memory", "4g") \
        

