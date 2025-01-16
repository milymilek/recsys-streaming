from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, from_unixtime, to_timestamp
from pyspark.sql.types import ArrayType, BooleanType, FloatType, LongType, StringType, StructField, StructType, TimestampType

from recsys_lakehouse.lakehouse.table import Table


@dataclass
class BooksReviewsTable(Table):
    table_name: str = "books_reviews"
    partition_col: str = "date_month"

    @property
    def schema(self):
        return StructType(
            [
                StructField("timestamp", TimestampType(), False),
                StructField("date_month", StringType(), False),
                StructField("user_id", StringType(), False),
                StructField("asin", StringType(), False),
                StructField("parent_asin", StringType(), False),
                StructField("helpful_vote", LongType(), False),
                StructField("verified_purchase", BooleanType(), False),
                StructField("title", StringType(), False),
                StructField("text", StringType(), False),
                StructField("rating", FloatType(), False),
            ]
        )

    def _process_columns(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("timestamp_iso", from_unixtime(col("timestamp") / 1000, "yyyy-MM-dd HH:mm:ss"))
            .drop("timestamp")
            .withColumn("timestamp", to_timestamp(col("timestamp_iso")))
            .withColumn("rating", col("rating").cast(FloatType()))
        )

    def _filter(self, df: DataFrame) -> DataFrame:
        return (
            df.filter(col("rating") <= 5.0)
            .filter(col("rating") >= 1.0)
            .filter(col("rating").isNotNull())
            .filter(col("asin").isNotNull())
            .filter(col("user_id").isNotNull())
            .filter(col("helpful_vote") >= 0)
        )

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: DataFrame) -> DataFrame:
        books_processed = self._process_columns(df)
        books_filtered = self._filter(books_processed)
        return self._only_schema_cols(books_filtered)


@dataclass
class BooksMetadataTable(Table):
    table_name: str = "books_metadata"
    partition_col: str = "main_category"

    @property
    def schema(self):
        return StructType(
            [
                StructField("parent_asin", StringType(), False),
                StructField("main_category", StringType(), False),
                StructField("title", StringType(), False),
                StructField("subtitle", StringType(), False),
                StructField("description", ArrayType(StringType()), False),
                StructField("categories", ArrayType(StringType()), False),
                StructField("price", FloatType(), False),
                StructField("average_rating", FloatType(), False),
                StructField("rating_number", LongType(), False),
                StructField(
                    "images",
                    ArrayType(
                        StructType(
                            [
                                StructField("hi_res", StringType(), True),
                                StructField("large", StringType(), True),
                                StructField("thumb", StringType(), True),
                                StructField("variant", StringType(), True),
                            ]
                        )
                    ),
                    False,
                ),
            ]
        )

    def _cast_types(self, df: DataFrame) -> DataFrame:
        return df.withColumn("price", col("price").cast(FloatType())).withColumn("average_rating", col("average_rating").cast(FloatType()))

    def _filter(self, df: DataFrame) -> DataFrame:
        return df.filter(col("parent_asin").isNotNull()).filter(col("average_rating") <= 5.0).filter(col("average_rating") >= 1.0)

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: DataFrame) -> DataFrame:
        books_metadata_filtered = self._filter(df)
        books_metadata_casted = self._cast_types(books_metadata_filtered)
        return self._only_schema_cols(books_metadata_casted)


silver_table_mapping = {BooksReviewsTable.table_name: BooksReviewsTable(), BooksMetadataTable.table_name: BooksMetadataTable()}
