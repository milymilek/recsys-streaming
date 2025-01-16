import logging

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse.gold._common import LayerConfig, parse_args, read_process_write_table
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    silver_layer_tables = ["books_reviews"]
    gold_layer_table = "user_features"
    read_process_write_table(spark, config, logger, silver_layer_tables, gold_layer_table)


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
