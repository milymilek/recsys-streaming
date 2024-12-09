# FROM apache/airflow:2.10.2-python3.11

# USER root

# RUN apt-get update && \ 
#     apt-get install -y curl gcc python3-dev openjdk-17-jdk && \
#     apt-get clean

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ENV PATH="$HOME/.local/bin:$PATH"
# ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-arm64"

# USER airflow

# COPY pyproject.toml pyproject.toml
# RUN uv pip install -r pyproject.toml


FROM apache/airflow:2.10.3-python3.12

USER root

RUN apt-get update

RUN apt-get install -y gcc python3-dev

RUN apt-get install -y openjdk-17-jdk

RUN apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-arm64

USER airflow

RUN pip install pyspark==3.5.3

RUN pip install apache-airflow==2.10.3

RUN pip install apache-airflow-providers-openlineage

RUN pip install apache-airflow-providers-google

RUN pip install apache-airflow-providers-apache-spark
