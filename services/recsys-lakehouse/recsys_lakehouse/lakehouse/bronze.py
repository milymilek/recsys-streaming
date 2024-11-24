from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, from_unixtime

from recsys_lakehouse.lakehouse.layers import Table


class BooksTable(Table):
    partition_col = "date"

    def partition_by(self, output_dir: Path):
        self._df = self._df.withColumn(self.partition_col, date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir))


class MetaBooksTable(Table):
    partition_col = "main_category"

    def partition_by(self, output_dir: Path):
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir))


class TableFactory:
    @staticmethod
    def get_table_object(df, table_name: str) -> Table:
        table_classes = {
            "Books": BooksTable,
            "meta_Books": MetaBooksTable,
        }
        c = table_classes.get(table_name)

        if c is None:
            raise ValueError(f"Table {table_name} not found.")

        return c(df)
