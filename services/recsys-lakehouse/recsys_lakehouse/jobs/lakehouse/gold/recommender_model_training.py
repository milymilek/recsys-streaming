import logging
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, rand, sum, when
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, TimestampType

from recsys_lakehouse.lakehouse import layers
from recsys_lakehouse.lakehouse.gold._common import LayerConfig, parse_args
from recsys_lakehouse.lakehouse.operator import TableOperator
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_mapping_df(df: DataFrame, col: str, mapping_suffix: str = "_index") -> DataFrame:
    if col not in df.columns:
        raise ValueError(f"Column {col} not in DataFrame")

    mapping_rdd = df.select(col).rdd.map(lambda row: row[col]).distinct().zipWithIndex()
    mapping_df = mapping_rdd.toDF([col, col + mapping_suffix])
    return mapping_df


def apply_mapping(df: DataFrame, **kwargs) -> DataFrame:
    for col, mapping_df in kwargs.items():
        mapping_col = next(iter(mapping_df.drop(col).columns), None)
        if mapping_col is None:
            raise ValueError(f"Column {col} not in mapping DataFrame")

        df = df.join(mapping_df, on=col, how="left")
        df = df.drop(col)
        df = df.withColumnRenamed(mapping_col, col)
    return df


def evaluate(model, df, evaluator):
    preds = model.transform(df)
    return evaluator.evaluate(preds)


def save_model_to_registry(model, model_registry_path: str):
    model.save((Path(model_registry_path) / "als_model").as_posix())


@log_wrapper(enter="Starting recommender model training.", exit="Recommender model training completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    operator = TableOperator(spark)
    gold = layers.Gold(dataset_name=config.dataset_name)

    reviews_df = operator.read_table(gold, table=gold.tables["reviews"])

    user_mapping = build_mapping_df(reviews_df, col="user_id")
    item_mapping = build_mapping_df(reviews_df, col="parent_asin")

    reviews_df_mapped = apply_mapping(reviews_df, user_id=user_mapping, parent_asin=item_mapping)

    (train_df, valid_df, test_df) = reviews_df_mapped.randomSplit([0.7, 0.2, 0.1])

    avg_rating = train_df.selectExpr("avg(rating) as avg_rating").first()["avg_rating"]
    baseline_predictions_train = train_df.withColumn("prediction", F.lit(avg_rating))
    baseline_predictions_valid = valid_df.withColumn("prediction", F.lit(avg_rating))

    als = ALS(
        maxIter=10, rank=16, regParam=0.01, userCol="user_id", itemCol="parent_asin", ratingCol="rating", coldStartStrategy="drop", seed=0
    )
    evaluator = RegressionEvaluator(metricName="rmse", predictionCol="prediction", labelCol="rating")

    pre_train_rmse = evaluator.evaluate(baseline_predictions_train)
    pre_valid_rmse = evaluator.evaluate(baseline_predictions_valid)

    model = als.fit(train_df)

    post_train_rmse = evaluate(model, train_df, evaluator)
    post_valid_rmse = evaluate(model, valid_df, evaluator)

    print(f"Pre Train RMSE (baseline): {pre_train_rmse} | Pre Validation RMSE (baseline): {pre_valid_rmse}")
    print(f"Post Train RMSE (ALS): {post_train_rmse} | Post Validation RMSE (ALS): {post_valid_rmse}")

    save_model_to_registry(model, model_registry_path=".datalake/model_registry")


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
