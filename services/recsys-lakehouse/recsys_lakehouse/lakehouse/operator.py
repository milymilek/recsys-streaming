from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from recsys_lakehouse.lakehouse.layers import Layer
from recsys_lakehouse.lakehouse.table import Table


class TableOperator:
    def __init__(self, spark: SparkSession):
        self._spark = spark

    def _table_path(self, layer: Layer, table: Table) -> Path:
        return layer.path / table.table_name

    def read_table(self, layer: Layer, table: Table) -> DataFrame:
        return self._spark.read.parquet(str(self._table_path(layer, table)))

    def write_table(self, df: DataFrame, table: Table, layer: Layer) -> None:
        df.write.mode("overwrite").partitionBy(table.partition_col).parquet(str(self._table_path(layer, table)))
