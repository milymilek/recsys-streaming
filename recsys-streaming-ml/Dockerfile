# The builder image, used to build the virtual environment
FROM python:3.10-buster as builder

RUN pip install poetry==1.6.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.10-buster as runtime

# Set venv envs
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Config Spark envs
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
    SPARK_HOME="/opt/spark"
RUN mkdir ${SPARK_HOME}

# install Java
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean;

# install Spark
RUN curl https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz -o spark-3.5.1-bin-hadoop3.tgz \
    && tar xvzf spark-3.5.1-bin-hadoop3.tgz --directory ${SPARK_HOME} --strip-components 1 \
    && rm -rf spark-3.5.1-bin-hadoop3.tgz

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY recsys_streaming_ml ./recsys_streaming_ml
COPY jars/* ./jars/*
COPY run.py ./run.py

# Suspend container so other tasks can run commands on the environment
CMD ["sleep", "infinity"]