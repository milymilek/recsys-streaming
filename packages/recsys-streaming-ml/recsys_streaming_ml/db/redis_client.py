import pandas as pd
from pyspark.sql import SparkSession
from redis import Redis
import pyspark


from recsys_streaming_ml.config import REDIS_HOST, REDIS_PORT


def send_recommendations_to_file(df: pyspark.sql.dataframe.DataFrame):
    df.toPandas().to_csv(".data/recommendations.csv")


def save_to_redis(partition):
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    for row in partition:
        redis_client.set(row.user_id, ','.join(row.top_k_asins))


def send_recommendations_to_redis(df: pyspark.sql.dataframe.DataFrame):
    print('Sending recommendations to redis...')
    df.rdd.foreachPartition(save_to_redis)


def get_recommendations():
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    all_keys = redis_client.keys('*')
    for key in all_keys:
        value = redis_client.get(key)
        print(f"{key}: {value}")


if __name__ == "__main__":
    get_recommendations()


# if __name__ == "__main__":
#     test = SparkSession.builder.appName("Redis Integration").getOrCreate().createDataFrame([
#         ("AFT3TBMJGYIJ4IUMPAICJYCDBUTQ", ["B08MKLHHBN", "B09F6GSRTH", "B099S5HPWB", "B09ZJ67MDK", "B0992LL6TG"]),
#         ("AGLQ7UWSBKN70IRDWURY25FZQ7AA", ["B08MKLHHBN", "B09F6GSRTH", "B099S5HPWB", "B09ZJ67MDK", "B0992LL6TG"]),
#     ], ["user_id", "top_k_asins"])
#     send_recommendations_to_redis(test)
#     get_recommendations()
