import pandas as pd
import pathlib
import json
import pickle
from sklearn.model_selection import TimeSeriesSplit

import pyspark.pandas as ps
from pyspark.sql.functions import rand

from recsys_streaming_ml.config import DATASET_FILE, TIMESTAMP_COL
from recsys_streaming_ml.data.utils import dump_feature_maps
from recsys_streaming_ml.db import mongo_db, read_df_from_mongo
from recsys_streaming_ml.spark.utils import spark


# def _build_map(df, col):
#     unique_vals = df[col].unique().to_numpy()
#     return dict(zip(unique_vals, range(len(unique_vals))))

# def _preprocess_data(df_ratings: pd.DataFrame, df_metadata: pd.DataFrame) -> pd.DataFrame:
#     #df = df_ratings.sort_values(by=TIMESTAMP_COL) # Dont sort by time as we want to include streamed data in training
#     df = df_ratings.merge(df_metadata, left_on='parent_asin', right_on='parent_asin')
#     df = df.dropna(subset=['store'])

#     #df.to_spark().write.csv((DATASET_FILE / "df_proc.csv").as_posix())

#     USER_ID_MAP = _build_map(df, 'user_id')
#     PARENT_ASIN_MAP = _build_map(df, 'parent_asin')
#     STORE_MAP = _build_map(df, 'store')
#     dump_feature_maps(user_id_map=USER_ID_MAP, parent_id_map=PARENT_ASIN_MAP, store_id_map=STORE_MAP)

#     df['user_id'] = df['user_id'].map(USER_ID_MAP)
#     df['parent_asin'] = df['parent_asin'].map(PARENT_ASIN_MAP)
#     df['store'] = df['store'].map(STORE_MAP)

#     #df.to_spark().write.csv((DATASET_FILE / "df_map.csv").as_posix())
#     dataframe = df.to_spark().toPandas()
#     DATASET_FILE.mkdir(parents=True, exist_ok=True)
#     dataframe.to_csv((DATASET_FILE / "dataset.csv").as_posix())


#     return df

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.types import IntegerType
import pandas as pd


def _build_map(spark_df, col):
    unique_vals = spark_df.select(col).distinct().rdd.map(lambda row: row[0]).collect()
    return {val: idx for idx, val in enumerate(unique_vals)}


def _preprocess_data(session, df_ratings, df_metadata, output_path):
    # Convert Pandas DataFrames to Spark DataFrames
    spark_df_ratings = session.createDataFrame(df_ratings)
    spark_df_metadata = session.createDataFrame(df_metadata)
    
    # Merge DataFrames on 'parent_asin'
    df = spark_df_ratings.join(spark_df_metadata, on='parent_asin')
    
    # Drop rows with null values in the 'store' column
    df_filtered = df.dropna(subset=['store'])
    
    # Build mappings for user_id, parent_asin, and store
    USER_ID_MAP = _build_map(df_filtered, 'user_id')
    PARENT_ASIN_MAP = _build_map(df_filtered, 'parent_asin')
    STORE_MAP = _build_map(df_filtered, 'store')
    
    # Dump feature maps (assuming the existence of the dump_feature_maps function)
    dump_feature_maps(user_id_map=USER_ID_MAP, parent_id_map=PARENT_ASIN_MAP, store_id_map=STORE_MAP)
    
    # Create broadcast variables for mappings
    user_id_map_broadcast = session.sparkContext.broadcast(USER_ID_MAP)
    parent_asin_map_broadcast = session.sparkContext.broadcast(PARENT_ASIN_MAP)
    store_map_broadcast = session.sparkContext.broadcast(STORE_MAP)
    
    # Define UDFs to map the IDs
    def map_user_id(user_id):
        return user_id_map_broadcast.value.get(user_id, -1)
    
    def map_parent_asin(parent_asin):
        return parent_asin_map_broadcast.value.get(parent_asin, -1)
    
    def map_store(store):
        return store_map_broadcast.value.get(store, -1)
    
    map_user_id_udf = session.udf.register("map_user_id", map_user_id, IntegerType())
    map_parent_asin_udf = session.udf.register("map_parent_asin", map_parent_asin, IntegerType())
    map_store_udf = session.udf.register("map_store", map_store, IntegerType())
    
    # Apply the UDFs to the DataFrame
    df_mapped = df_filtered.withColumn("user_id", map_user_id_udf(col("user_id"))) \
           .withColumn("parent_asin", map_parent_asin_udf(col("parent_asin"))) \
           .withColumn("store", map_store_udf(col("store")))

    # Save the resulting DataFrame to CSV
    #df.coalesce(1).write.option("header", "true").csv(output_path)
    df_mapped.coalesce(1).write.format("csv").mode('overwrite').option("header", "true").save(output_path)
    
    return df


def _dump_data(data: dict, path: pathlib.Path) -> None:
    ...

def process_data(df_rating_stream=None):
    _names_list = ["ratings", "metadata"]
    session = spark()

    #print(df_rating_stream)
    #print("+"*20)
    #df_ratings_stream = pd.DataFrame() #read_stream(...)   df_rating_stream
    df_ratings_historical = read_df_from_mongo(db=mongo_db, collection=_names_list[0])
    #print(df_ratings_historical)
    #print("+"*20)
    df_ratings = pd.concat([df_rating_stream, df_ratings_historical])
    df_metadata = read_df_from_mongo(db=mongo_db, collection=_names_list[1])

    #psdf_ratings = ps.from_pandas(df_ratings)
    #psdf_metadata = ps.from_pandas(df_metadata)

    #psdf_ratings.to_spark().write.csv((DATASET_FILE / "tr/train_data.csv").as_posix())
    #psdf_metadata.to_spark().write.csv((DATASET_FILE / "v/df_validation.csv").as_posix())

    #filtered_data = _preprocess_data(psdf_ratings, psdf_metadata)
    _preprocess_data(session, df_ratings, df_metadata, output_path=DATASET_FILE.as_posix())

    #session.stop()


def run():
    print("SCRIPT: Process data - START")

    process_data()

    print("SCRIPT: Process data - END")