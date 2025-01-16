from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

with DAG(
    "lakehouse-recommendation",
    start_date=days_ago(1),
    schedule_interval=None,
) as dag:
    recommender_model_inference = SparkSubmitOperator(
        task_id="recommender_model_inference",
        application="recsys_lakehouse/jobs/lakehouse/gold/recommender_model_inference.py",
        conn_id="spark-conn",
        dag=dag,
    )
