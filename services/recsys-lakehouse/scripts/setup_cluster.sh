#!/bin/bash

# Defaults
WORKERS=1
WORKER_RAM_POOL=12
WORKER_CPU_POOL=6

OPTSTRING=":n:"
while getopts ${OPTSTRING} opt; do
  case ${opt} in
    n) WORKERS=$OPTARG ;;
  esac
done


docker build -t spark-base .

docker network create recsys-streaming-net >/dev/null 2>&1 || true

# Master
docker rm -f spark-master >/dev/null 2>&1 || true
docker run -d \
    --name spark-master \
    --net recsys-streaming-net \
    -p 9090:8080 \
    -p 7077:7077 \
    -v "$(pwd)/.datalake:/opt/bitnami/spark/.datalake" \
    -v "$(pwd)/recsys_lakehouse:/opt/bitnami/spark/recsys_lakehouse" \
    spark-base \
    /bin/bash -c "bin/spark-class org.apache.spark.deploy.master.Master"

# Worker
ram_per_worker=$(($WORKER_RAM_POOL / $WORKERS))
cpu_per_worker=$(($WORKER_CPU_POOL / $WORKERS))
echo -e "RAM per worker: $ram_per_worker \nCPU per worker: $cpu_per_worker"

docker rm -f $(docker ps -a | grep 'spark-worker-' | awk '{print $1}') >/dev/null 2>&1 || true
for i in $(seq 1 $WORKERS); do
  docker run -d \
      --name spark-worker-$i \
      --net recsys-streaming-net \
      -v "$(pwd)/.datalake:/opt/bitnami/spark/.datalake" \
      -v "$(pwd)/recsys_lakehouse:/opt/bitnami/spark/recsys_lakehouse" \
      -e SPARK_MODE=worker \
      -e SPARK_WORKER_CORES="${cpu_per_worker}" \
      -e SPARK_WORKER_MEMORY="${ram_per_worker}G"\
      -e SPARK_MASTER_URL=spark://spark-master:7077 \
      spark-base \
      /bin/bash -c "echo '123' && bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077"
done