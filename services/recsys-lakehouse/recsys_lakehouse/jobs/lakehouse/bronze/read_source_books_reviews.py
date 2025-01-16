import logging

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse.bronze._common import LayerConfig, parse_args, read_process_write_table
from recsys_lakehouse.spark import spark_builder
from recsys_lakehouse.utils import log_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@log_wrapper(enter="Starting ingestion to bronze layer.", exit="Bronze layer ingestion completed successfully.")
def main(spark: SparkSession, config) -> None:
    logger.info("Reading `%s`...", "books_reviews")
    read_process_write_table(spark, config, "books_reviews")


if __name__ == "__main__":
    args = parse_args()
    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        raw_data_source=args.raw_data_source,
        dataset_name=args.dataset_name,
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
