from datetime import datetime

import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

dag = DAG(dag_id="dummy_cron_job", default_args={"owner": "MH", "start_date": datetime.now()}, schedule_interval="* * * * *")

start = PythonOperator(task_id="start", python_callable=lambda: print("Jobs started"), dag=dag)

python_job = SparkSubmitOperator(task_id="counter", conn_id="spark-conn", application="jobs/python/dummy_count_job.py", dag=dag)

end = PythonOperator(task_id="end", python_callable=lambda: print("Jobs completed successfully"), dag=dag)

start >> python_job >> end
