from abc import ABC, abstractmethod
from pathlib import Path

from pyspark.sql import DataFrame

from recsys_lakehouse.lakehouse import bronze, gold, silver
from recsys_lakehouse.lakehouse.raw_data_source import RawDataSource
from recsys_lakehouse.lakehouse.table import Table


class Layer(ABC):
    def __init__(self, dataset_name: str, datalake_uri: str = ".datalake"):
        self._datalake_uri = datalake_uri
        self._dataset_name = dataset_name

    @property
    def base_path(self) -> Path:
        return Path(self._datalake_uri) / self.layer_path

    @property
    @abstractmethod
    def layer_path(self) -> Path:
        pass

    @property
    def path(self) -> Path:
        return self.base_path / self._dataset_name


class Raw(Layer):
    def __init__(self, source: RawDataSource, dataset_name: str):
        super().__init__(dataset_name)
        self._source = source

    @property
    def layer_path(self) -> Path:
        return Path("raw")

    def read_source(self) -> dict[str, DataFrame]:
        return self._source.read()


class Bronze(Layer):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)

    @property
    def layer_path(self) -> Path:
        return Path("bronze")

    @property
    def tables(self) -> dict[str, Table]:
        return bronze.bronze_table_mapping


class Silver(Layer):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)

    @property
    def layer_path(self) -> Path:
        return Path("silver")

    @property
    def tables(self) -> dict[str, Table]:
        return silver.silver_table_mapping


class Gold(Layer):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)

    @property
    def layer_path(self) -> Path:
        return Path("gold")

    @property
    def tables(self) -> dict[str, Table]:
        return gold.gold_table_mapping
