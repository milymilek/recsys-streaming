from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

with DAG(
    "lakehouse-historic",
    start_date=days_ago(1),
    schedule_interval=None,
    # params={
    #     "app_name": Param(default="Bronze Layer - Ingestion", type="string", minLength=1, maxLength=255),
    #     "error_log_level": Param(default="ERROR", type="string", enum=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    #     "raw_data_source": Param(default="JSONDataSource", type="string", minLength=1, maxLength=100),
    #     "dataset_name": Param(default="amazon_books_sample10000", type="string", minLength=1, maxLength=255),
    # },
) as dag:
    # bronze_start = PythonOperator(task_id="bronze_start", python_callable=lambda: print("Bronze layer processing begins..."), dag=dag)

    # read_source_books_reviews = SparkSubmitOperator(
    #     task_id="read_source_books_reviews",
    #     application="recsys_lakehouse/jobs/lakehouse/bronze/read_source_books_reviews.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )
    # read_source_books_metadata = SparkSubmitOperator(
    #     task_id="read_source_books_metadata",
    #     application="recsys_lakehouse/jobs/lakehouse/bronze/read_source_books_metadata.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # bronze_end = PythonOperator(task_id="bronze_end", python_callable=lambda: print("Bronze layer finished..."), dag=dag)

    # silver_start = PythonOperator(task_id="silver_start", python_callable=lambda: print("Silver layer processing begins..."), dag=dag)

    # clean_bronze_books_reviews = SparkSubmitOperator(
    #     task_id="clean_bronze_books_reviews",
    #     application="recsys_lakehouse/jobs/lakehouse/silver/clean_bronze_books_reviews.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )
    # clean_bronze_books_metadata = SparkSubmitOperator(
    #     task_id="clean_bronze_books_metadata",
    #     application="recsys_lakehouse/jobs/lakehouse/silver/clean_bronze_books_metadata.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # silver_end = PythonOperator(task_id="silver_end", python_callable=lambda: print("Silver layer finished..."), dag=dag)

    # gold_start = PythonOperator(task_id="gold_start", python_callable=lambda: print("Gold layer processing begins..."), dag=dag)

    # build_gold_reviews_table = SparkSubmitOperator(
    #     task_id="build_gold_reviews_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_gold_reviews_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # reviews_table_vis = SparkSubmitOperator(
    #     task_id="reviews_table_vis",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/reviews_table_vis.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    recommender_model_training = SparkSubmitOperator(
        task_id="recommender_model_training",
        application="recsys_lakehouse/jobs/lakehouse/gold/recommender_model_training.py",
        conn_id="spark-conn",
        dag=dag,
    )

    # build_explicit_content_items_table = SparkSubmitOperator(
    #     task_id="build_explicit_content_items_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_explicit_content_items_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # build_user_review_moderation_table = SparkSubmitOperator(
    #     task_id="build_user_review_moderation_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_user_review_moderation_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # build_user_features_table = SparkSubmitOperator(
    #     task_id="build_user_features_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_user_features_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # build_item_features_table = SparkSubmitOperator(
    #     task_id="build_item_features_table",
    #     application="recsys_lakehouse/jobs/lakehouse/gold/build_item_features_table.py",
    #     conn_id="spark-conn",
    #     dag=dag,
    # )

    # gold_end = PythonOperator(task_id="gold_end", python_callable=lambda: print("Gold layer finished..."), dag=dag)

    # bronze_start >> [read_source_books_reviews, read_source_books_metadata] >> bronze_end
    # bronze_end >> silver_start
    # silver_start >> [clean_bronze_books_reviews, clean_bronze_books_metadata] >> silver_end
    # silver_end >> gold_start
    # (
    #     gold_start
    #     >> [
    #         build_gold_reviews_table,
    #         build_user_features_table,
    #         build_item_features_table,
    #         build_user_review_moderation_table,
    #         build_explicit_content_items_table,
    #     ]
    #     >> gold_end
    # )

    # build_gold_reviews_table >> reviews_table_vis
