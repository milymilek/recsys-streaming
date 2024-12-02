from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession


class RawDataSource(ABC):
    def __init__(self, spark: SparkSession):
        self.spark = spark

    @abstractmethod
    def read(self) -> dict[str, Any]:
        pass


class JSONDataSource(RawDataSource):
    def __init__(self, base_path: Path, tables: dict[str, Any], spark: SparkSession):
        super().__init__(spark)
        self._base_path = base_path
        self._tables = tables
        self._validate_files_exist()

    def _validate_files_exist(self) -> None:
        # files = self.get_files()
        # assert files, "No files found."
        # missing_files = ...
        # assert not missing_files, f"Missing files: {missing_files}"
        ...

    def read(self) -> dict[str, Any]:
        return {table: self.spark.read.json(str(self._base_path / file)) for table, file in self._tables.items()}

    @property
    def files(self) -> list[str]:
        return ["Books.jsonl", "meta_Books.jsonl"]


class StreamDataSource(RawDataSource):
    def __init__(self, tables: dict[str, Any], spark: SparkSession):
        super().__init__(spark)
        self._tables = tables

    def read(self) -> Any:
        return {table: self.spark.readStream.format("rate").load() for table, stream in self._tables.items()}
