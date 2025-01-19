from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule


def decide_to_run_sample_data(**kwargs):
    params = kwargs["params"]
    if params.get("sample_data", False):
        return "sample_books_data"
    return "skip_sample_books_data"


with DAG(
    "lakehouse-historic",
    start_date=days_ago(1),
    schedule_interval=None,
    params={
        "dataset_name": Param(default="amazon_books_from_2023-06", type="string", minLength=1, maxLength=255),
        "sample_data": Param(default=False, type="boolean"),
    },
) as dag:
    # raw_start = PythonOperator(task_id="raw_start", python_callable=lambda: print("Raw data preparation begins..."), dag=dag)

    # decide_task = BranchPythonOperator(
    #     task_id="decide_task",
    #     python_callable=decide_to_run_sample_data,
    #     provide_context=True,
    # )

    # sample_books_data = SparkSubmitOperator(
    #     task_id="sample_books_data", application="recsys_lakehouse/jobs/lakehouse/raw/sample_books_data.py", conn_id="spark-conn", dag=dag
    # )

    # skip_sample_books_data = PythonOperator(
    #     task_id="skip_sample_books_data",
    #     python_callable=lambda: print("Skipping sample books data..."),
    # )

    # raw_end = PythonOperator(
    #     task_id="raw_end",
    #     python_callable=lambda: print("Raw data preparation finished..."),
    #     dag=dag,
    #     trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    # )

    # bronze_start = PythonOperator(task_id="bronze_start", python_callable=lambda: print("Bronze layer processing begins..."), dag=dag)

    # read_source_books_reviews = SparkSubmitOperator(
    #     task_id="read_source_books_reviews",
    #     application="recsys_lakehouse/jobs/lakehouse/bronze/read_source_books_reviews.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )
    # read_source_books_metadata = SparkSubmitOperator(
    #     task_id="read_source_books_metadata",
    #     application="recsys_lakehouse/jobs/lakehouse/bronze/read_source_books_metadata.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # bronze_end = PythonOperator(task_id="bronze_end", python_callable=lambda: print("Bronze layer finished..."), dag=dag)

    # silver_start = PythonOperator(task_id="silver_start", python_callable=lambda: print("Silver layer processing begins..."), dag=dag)

    # clean_bronze_books_reviews = SparkSubmitOperator(
    #     task_id="clean_bronze_books_reviews",
    #     application="recsys_lakehouse/jobs/lakehouse/silver/clean_bronze_books_reviews.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )
    # clean_bronze_books_metadata = SparkSubmitOperator(
    #     task_id="clean_bronze_books_metadata",
    #     application="recsys_lakehouse/jobs/lakehouse/silver/clean_bronze_books_metadata.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # silver_end = PythonOperator(task_id="silver_end", python_callable=lambda: print("Silver layer finished..."), dag=dag)

    # gold_start = PythonOperator(task_id="gold_start", python_callable=lambda: print("Gold layer processing begins..."), dag=dag)

    # build_gold_reviews_table = SparkSubmitOperator(
    #     task_id="build_gold_reviews_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_gold_reviews_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    reviews_table_vis = SparkSubmitOperator(
        task_id="reviews_table_vis",
        application="recsys_lakehouse/jobs/lakehouse/gold/reviews_table_vis.py",
        conn_id="spark-conn",
        dag=dag,
        application_args=["--dataset_name", "{{ params.dataset_name }}"],
    )

    item_features_table_vis = SparkSubmitOperator(
        task_id="item_features_table_vis",
        application="recsys_lakehouse/jobs/lakehouse/gold/item_features_table_vis.py",
        conn_id="spark-conn",
        dag=dag,
        application_args=["--dataset_name", "{{ params.dataset_name }}"],
    )

    # recommender_model_training = SparkSubmitOperator(
    #     task_id="recommender_model_training",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/recommender_model_training.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # build_explicit_content_items_table = SparkSubmitOperator(
    #     task_id="build_explicit_content_items_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_explicit_content_items_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # build_user_review_moderation_table = SparkSubmitOperator(
    #     task_id="build_user_review_moderation_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_user_review_moderation_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # build_user_features_table = SparkSubmitOperator(
    #     task_id="build_user_features_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_user_features_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # build_item_features_table = SparkSubmitOperator(
    #     task_id="build_item_features_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_item_features_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    #     application_args=["--dataset_name", "{{ params.dataset_name }}"],
    # )

    # gold_end = PythonOperator(task_id="gold_end", python_callable=lambda: print("Gold layer finished..."), dag=dag)

    # raw_start >> decide_task
    # decide_task >> [sample_books_data, skip_sample_books_data]
    # [sample_books_data, skip_sample_books_data] >> raw_end

    # raw_end >> bronze_start

    # bronze_start >> [read_source_books_reviews, read_source_books_metadata] >> bronze_end

    # bronze_end >> silver_start

    # silver_start >> [clean_bronze_books_reviews, clean_bronze_books_metadata] >> silver_end

    # silver_end >> gold_start

    # (
    #     gold_start
    #     >> [
    #         build_user_features_table,
    #         build_item_features_table,
    #         build_user_review_moderation_table,
    #         build_explicit_content_items_table,
    #     ]
    #     >> gold_end
    # )
    # gold_start >> build_gold_reviews_table >> [recommender_model_training, reviews_table_vis] >> gold_end
