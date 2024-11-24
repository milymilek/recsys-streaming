FROM bitnami/spark:3.5.3

WORKDIR /opt/bitnami/spark

USER root

ENV SPARK_SUBMIT_ARGS="--packages io.delta:delta-core_2.12:2.4.0"

RUN python3 -m virtualenv ".venv"
ENV PATH="/opt/bitnami/spark/.venv/bin:${PATH}"

COPY ./recsys_lakehouse/ recsys_lakehouse/
