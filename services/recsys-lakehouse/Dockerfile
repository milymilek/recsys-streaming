FROM bitnami/spark:3.5.3

WORKDIR /opt/bitnami/spark

USER root

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install curl -y

ENV SPARK_SUBMIT_ARGS="--packages io.delta:delta-core_2.12:2.4.0"

RUN python3 -m virtualenv ".venv"
ENV PATH="/opt/bitnami/spark/.venv/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env

RUN pip install numpy

COPY ./recsys_lakehouse/ recsys_lakehouse/
