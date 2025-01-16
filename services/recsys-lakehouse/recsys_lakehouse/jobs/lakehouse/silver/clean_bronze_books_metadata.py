import logging

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse.silver._common import LayerConfig, clean_bronze_write_silver, parse_args
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession, config: LayerConfig) -> None:
    clean_bronze_write_silver(spark, config, logger, table_name="books_metadata")


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
