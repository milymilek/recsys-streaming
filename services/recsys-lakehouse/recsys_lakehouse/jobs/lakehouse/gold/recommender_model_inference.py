import logging
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
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


@log_wrapper(enter="Starting recommender model inference.", exit="Recommender model inference completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    model_path = ".datalake/model_registry/als_model"
    als_model = ALSModel.load(model_path)

    user_df = spark.createDataFrame([(1,), (2,), (3,), (4,)], ["user_id"])
    als_model.recommendForUserSubset(user_df, numItems=10).show()


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
