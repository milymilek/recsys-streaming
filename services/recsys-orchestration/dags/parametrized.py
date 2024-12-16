from airflow import DAG
from airflow.models.param import Param
from airflow.utils.dates import days_ago

from recsys_lakehouse.jobs.lakehouse.bronze import main, LayerConfig
from recsys_lakehouse.spark import spark_builder
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


def run_main(**kwargs):
    params = kwargs['params']
    config = LayerConfig(
        app_name=params['app_name'],
        error_log_level=params['error_log_level'],
        raw_data_source=params['raw_data_source'],
        dataset_name=params['dataset_name'],
    )

    with spark_builder(config.app_name, config.error_log_level) as spark:
        main(spark, config)

with DAG(
    'parameterized_dag',
    start_date=days_ago(1),
    schedule_interval=None,
    params={
        "app_name": Param(
            default="Bronze Layer - Ingestion",
            type="string",
            minLength=1,
            maxLength=255
        ),
        "error_log_level": Param(
            default="ERROR",
            type="string",
            enum=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ),
        "raw_data_source": Param(
            default="JSONDataSource",
            type="string",
            minLength=1,
            maxLength=100
        ),
        "dataset_name": Param(
            default="amazon_books_sample10000",
            type="string",
            minLength=1,
            maxLength=255
        ),
    },
) as dag:
    task = SparkSubmitOperator(
        task_id='run_bronze_layer_ingestion',
        application='dags/bronze_layer_ingestion.py',
        # application_args=[
        #     "--app_name", "{{ params.app_name }}",
        #     "--error_log_level", "{{ params.error_log_level }}",
        #     "--raw_data_source", "{{ params.raw_data_source }}",
        #     "--dataset_name", "{{ params.dataset_name }}"
        # ],
        conn_id="spark-conn"
    )
