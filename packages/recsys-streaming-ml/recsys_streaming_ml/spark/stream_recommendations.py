from itertools import chain
import numpy as np
import pandas as pd
import torch

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array, col, create_map, lit, rank, collect_list, from_json
)
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, StructType, StructField, StringType
from pyspark.ml.functions import predict_batch_udf

from recsys_streaming_ml.spark.utils import spark, spark_structured_streaming
from recsys_streaming_ml.model.utils import load_model_from_db, build_input_tensor
from recsys_streaming_ml.data.utils import load_feature_maps, read_item_feature_store, build_reverse_feature_maps
from recsys_streaming_ml.db import mongo_db
from recsys_streaming_ml.db.redis_client import send_recommendations_to_file, send_recommendations_to_redis
from recsys_streaming_ml.config import DATA_DIR, KAFKA_BROKER_URL, RECOMMENDATIONS_TOPIC, TRAINING_OFFSET, \
    EVALUATE_OFFSET


def process_data(
        df: pyspark.sql.DataFrame,
        item_feature_store: pyspark.sql.DataFrame,
        user_id_mapping: dict[str, int]
) -> pyspark.sql.dataframe.DataFrame:
    """
    Process the DataFrame by mapping user_ids using the provided dictionary.
    """
    mapping_expr = create_map([lit(x) for x in chain(*user_id_mapping.items())])

    processed_df = df.withColumn("map_user_id", mapping_expr[col("user_id")])
    processed_df = processed_df.crossJoin(item_feature_store)
    processed_df = processed_df.select("map_user_id", "parent_asin", "store_id")

    return processed_df


def predict_batch_fn(device='cpu'):
    # load model from checkpoint
    import torch
    from recsys_streaming_ml.db import mongo_db
    model = load_model_from_db(mongo_db, device)

    # define predict function in terms of numpy arrays
    def predict(inputs: np.ndarray) -> np.ndarray:
        torch_inputs = build_input_tensor(inputs).to(device)
        outputs = model(torch_inputs)
        return outputs.cpu().detach().numpy()

    return predict


def get_ranked_topk_predictions(df, k=5):
    window = Window.partitionBy("map_user_id").orderBy(col("predicted_rating").desc())

    # Add a rank column to rank the rows within each partition by 'sum'
    ranked_predictions = df.withColumn("rank", rank().over(window))

    # Filter to keep only the top 5 'asin' values for each 'map_user_id'
    top_k = ranked_predictions.filter(col("rank") <= k)

    return top_k


def remap_entities(df, user_id_mapping, asin_mapping):
    mapping_expr_user = create_map([lit(x) for x in chain(*user_id_mapping.items())])
    mapping_expr_asin = create_map([lit(x) for x in chain(*asin_mapping.items())])

    df = df.withColumn("user_id", mapping_expr_user[col("map_user_id")])
    df = df.withColumn("asin", mapping_expr_asin[col("parent_asin")])

    return df.select("user_id", "asin", "rank")


def list_recommendations(df):
    # Aggregate the top k 'asin' values into a list for each 'map_user_id'
    result = df.groupBy("user_id").agg(collect_list("asin").alias("top_k_asins"))
    return result


def evaluate(users_df, batch_id):
    session: SparkSession = spark()
    print(f"Showing data for batch: {batch_id}")
    users_df.show()

    feature_maps: dict[str, int] = load_feature_maps()
    reverse_feature_maps: dict[int, str] = build_reverse_feature_maps(feature_maps)
    item_feature_store: pyspark.sql.DataFrame = session.createDataFrame(read_item_feature_store(mongo_db, feature_maps))

    processed_df = process_data(users_df, item_feature_store, feature_maps['user_id_map'])

    predict_udf = predict_batch_udf(
        predict_batch_fn,
        return_type=FloatType(),
        batch_size=128,
        input_tensor_shapes=[[3]]
    )

    predictions = processed_df.withColumn("predicted_rating",
                                          predict_udf(array("map_user_id", "parent_asin", "store_id")))
    predictions.show()

    ranked_topk = get_ranked_topk_predictions(predictions)
    remapped_ranked_topk = remap_entities(ranked_topk, reverse_feature_maps['user_id_map'],
                                          reverse_feature_maps['parent_id_map'])
    recommendation_lists = list_recommendations(remapped_ranked_topk)

    send_recommendations_to_redis(recommendation_lists)



def main():
    session: SparkSession = spark_structured_streaming()

    schema = StructType([StructField("user_id", StringType(), False)])

    kafkaStream = session.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER_URL) \
        .option("subscribe", RECOMMENDATIONS_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    dataStream = kafkaStream.select(
        from_json(col("value").cast("string"), schema).alias("jsonData")
    ).select("jsonData.*")

    query = dataStream.writeStream \
        .foreachBatch(evaluate) \
        .trigger(processingTime=EVALUATE_OFFSET) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    # consume()
    main()
