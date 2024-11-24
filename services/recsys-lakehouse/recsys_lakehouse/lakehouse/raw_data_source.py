from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession


class RawDataSource(ABC):
    def __init__(self, spark: SparkSession):
        self.spark = spark

    @abstractmethod
    def read(self) -> Any:
        pass


class JSONDataSource(RawDataSource):
    def __init__(self, spark: SparkSession):
        super().__init__(spark)

    def read(self) -> Any:
        return self.spark.read.json(str(None))


class StreamDataSource(RawDataSource):
    def __init__(self, spark: SparkSession):
        super().__init__(spark)
