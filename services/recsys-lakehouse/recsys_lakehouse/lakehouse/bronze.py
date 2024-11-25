from pathlib import Path

from pyspark.sql.functions import col, date_format, from_unixtime

from recsys_lakehouse.lakehouse.layers import Table


class BooksTable(Table):
    table_name = "books"
    partition_col = "date"

    def partition_by(self, output_dir: Path):
        self._df = self._df.withColumn(self.partition_col, date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir / self.table_name))


class MetaBooksTable(Table):
    table_name = "meta_books"
    partition_col = "main_category"

    def partition_by(self, output_dir: Path):
        self._df.write.mode("overwrite").partitionBy(self.partition_col).parquet(str(output_dir / self.table_name))


bronze_table_mapping = {BooksTable.table_name: BooksTable, MetaBooksTable.table_name: MetaBooksTable}
