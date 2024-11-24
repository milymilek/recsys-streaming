import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from recsys_lakehouse.lakehouse.raw_data_source import RawDataSource


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
    pass


class Gold(Layer):
    pass


class LayerOperator:
    def __init__(self, read_layer: Layer, write_layer: Layer, raw="raw"):
        self._raw = raw
        self._read_layer = read_layer
        self._write_layer = write_layer

    @property
    def base_path(self) -> Path:
        return Path(".datalake")

    def read_files(self) -> list[Path]:
        return list((self.base_path / self._raw).iterdir())

    def read_path(self, file_name: str) -> Path:
        return self.base_path / self._read_layer.value / file_name

    def write_path(self, file_name: str) -> Path:
        return self.base_path / self._write_layer.value / f"amazon_books/data_source=http_github/{file_name}"


class Table:
    def __init__(self, df):
        self._df = df

    @abstractmethod
    def partition_by(self, output_dir: Path):
        pass
