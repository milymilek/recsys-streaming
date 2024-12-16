import argparse

from recsys_lakehouse.jobs.lakehouse.bronze import main, LayerConfig
from recsys_lakehouse.spark import spark_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bronze Layer Configuration")
    parser.add_argument("--app_name", type=str, default="Bronze Layer - Ingestion")
    parser.add_argument("--error_log_level", type=str, default="ERROR")
    parser.add_argument("--raw_data_source", type=str, default="JSONDataSource")
    parser.add_argument("--dataset_name", type=str, required=False, default="amazon_books_sample10000")
    args = parser.parse_args()

    return args

# def main(app_name, error_log_level, raw_data_source, dataset_name):
#     config = LayerConfig(
#         app_name=app_name,
#         error_log_level=error_log_level,
#         raw_data_source=raw_data_source,
#         dataset_name=dataset_name,
#     )
#
#     with spark_builder(config.app_name, config.error_log_level) as spark:
#         print(f"Starting ingestion with dataset: {dataset_name}")
#         main(spark, config)

if __name__ == "__main__":
    args = parse_args()

    config = LayerConfig(
        app_name=args.app_name,
        error_log_level=args.error_log_level,
        raw_data_source=args.raw_data_source,
        dataset_name=args.dataset_name,
    )

    print(args)

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)
