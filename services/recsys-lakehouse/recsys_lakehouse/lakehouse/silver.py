from pathlib import Path

from pyspark.sql.functions import col, date_format, from_unixtime
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from recsys_lakehouse.lakehouse.layers import Table


class BooksReviewsTable(Table):
    table_name = "books_reviews"

    @property
    def schema(self):
        return StructType(
            [
                StructField("dates", StringType(), False),
                StructField("user_id", StringType(), False),
                StructField("asin", StringType(), False),
                StructField("rating", FloatType(), False),
            ]
        )

    def process(self):
        self._df = self._df.withColumn("dates", from_unixtime(col("timestamp") / 1000, "yyyy-MM-dd HH:mm:ss")).withColumn(
            "rating", col("rating").cast(FloatType())
        )


class BooksMetadataTable(Table):
    table_name = "books_metadata"

    @property
    def schema(self):
        return StructType(
            [
                StructField("parent_asin", StringType(), False),
            ]
        )

    def process(self): ...


silver_table_mapping = {BooksReviewsTable.table_name: BooksReviewsTable, BooksMetadataTable.table_name: BooksMetadataTable}
