FROM apache/airflow:2.10.2-python3.11

USER root

RUN apt-get update && \ 
    apt-get install -y curl gcc python3-dev openjdk-17-jdk && \
    apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set JAVA_HOME environment variable
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-arm64" \
    PATH="$HOME/.local/bin:$PATH"

USER airflow

COPY pyproject.toml /opt/airflow/pyproject.toml

RUN uv pip install -r /opt/airflow/pyproject.toml

ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"