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
        read_table = self._spark.read
        if table.schema is not None:
            read_table = read_table.schema(table.schema)
        return read_table.parquet(str(self._table_path(layer, table)))

    def write_table(self, df: DataFrame, table: Table, layer: Layer) -> None:
        write_fn = df.write.mode("overwrite")
        if table.partition_col:
            write_fn = write_fn.partitionBy(table.partition_col)
        write_fn.parquet(str(self._table_path(layer, table)))
