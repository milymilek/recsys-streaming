from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, date_format, from_unixtime

from recsys_lakehouse.lakehouse.table import Table


@dataclass
class BooksReviewsTable(Table):
    table_name: str = "books_reviews"
    partition_col: str = "date_month"

    def process(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self.partition_col, date_format(from_unixtime(col("timestamp") / 1000), "yyyy-MM"))


@dataclass
class BooksMetadataTable(Table):
    table_name: str = "books_metadata"
    partition_col: str = "main_category"


bronze_table_mapping = {BooksReviewsTable.table_name: BooksReviewsTable(), BooksMetadataTable.table_name: BooksMetadataTable()}
