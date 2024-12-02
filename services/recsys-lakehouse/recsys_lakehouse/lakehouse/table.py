from abc import abstractmethod
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


@dataclass
class Table:
    table_name: str
    partition_col: str

    @property
    def schema(self) -> StructType | None:
        return None

    def process(self, df: DataFrame) -> DataFrame:
        return df
