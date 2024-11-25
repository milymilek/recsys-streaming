import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from pyspark.sql import SparkSession

from recsys_lakehouse.lakehouse.raw_data_source import RawDataSource


class Table:
    def __init__(self, df):
        self._df = df

    @property
    def schema(self): ...

    @abstractmethod
    def partition_by(self, output_dir: Path):
        pass


class Layer(ABC):
    def __init__(self):
        self._datalake_uri = ".datalake"

    @property
    @abstractmethod
    def layer_path(self) -> Path:
        pass

    @property
    def base_path(self) -> Path:
        return Path(self._datalake_uri) / self.layer_path


class Raw(Layer):
    def __init__(self, source: RawDataSource, path: Path):
        super().__init__()
        self._source = source
        self._path = path

    @property
    def layer_path(self) -> Path:
        return Path("raw")

    @property
    def path(self) -> Path:
        return self.base_path / self._path

    def get_files(self):
        expected_files = self._source.expected_files
        files = list(self.path.iterdir())
        assert all(file.name in expected_files for file in files), f"Missing files: {expected_files}"
        return files


class Bronze(Layer):
    def __init__(self, path: Path):
        super().__init__()
        self._path = path

    @property
    def layer_path(self) -> Path:
        return Path("bronze")

    @property
    def path(self) -> Path:
        return self.base_path / self._path


class Silver(Layer):
    def __init__(self, path: Path, spark: SparkSession):
        super().__init__()
        self._path = path
        self._spark = spark

    @property
    def layer_path(self) -> Path:
        return Path("silver")

    @property
    def path(self) -> Path:
        return self.base_path / self._path

    def write_table(self, df, table: Table):
        columns_in_schema = [field.name for field in table.schema.fields]
        df_s = df.select(*columns_in_schema)

        df_enforced_schema = self._spark.createDataFrame(df_s.rdd, table.schema)
        df_enforced_schema.write.mode("overwrite").parquet(str(self.path / table.table_name))


class Gold(Layer):
    pass


class TableFactory:
    @staticmethod
    def get_table_object(df, table_name: str, table_mapping: dict[str, type[Table]]) -> Table:
        c = table_mapping.get(table_name)

        if c is None:
            raise ValueError(f"Table {table_name} not found.")

        return c(df)
