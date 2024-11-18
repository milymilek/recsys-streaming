FROM apache/airflow:2.10.2-python3.11

USER root

RUN apt-get update && \ 
    apt-get install -y curl gcc python3-dev openjdk-17-jdk && \
    apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="$HOME/.local/bin:$PATH"
ENV JAVA_HOM="/usr/lib/jvm/java-17-openjdk-arm64"

USER airflow

COPY pyproject.toml pyproject.toml
RUN uv pip install -r pyproject.toml
