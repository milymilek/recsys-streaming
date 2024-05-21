import pandas as pd

import pyspark


def send_recommendations_to_redis(df: pyspark.sql.dataframe.DataFrame):
    df.toPandas().to_csv(".data/recommendations.csv")