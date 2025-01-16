from dataclasses import dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_unixtime, to_timestamp
from pyspark.sql.types import ArrayType, BooleanType, FloatType, LongType, StringType, StructField, StructType, TimestampType

from recsys_lakehouse.lakehouse.silver.silver import BooksMetadataTable, BooksReviewsTable
from recsys_lakehouse.lakehouse.table import Table


@dataclass
class ExplicitContentItemsTable(Table):
    table_name: str = "explicit_content_items"
    partition_col: str = ""
    parent_tables: tuple[str] = (BooksMetadataTable().table_name,)

    @property
    def schema(self):
        return StructType(
            [
                StructField("parent_asin", StringType(), False),
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
                StructField("title", BooleanType(), False),
                StructField("description", ArrayType(StringType()), False),
            ]
        )

    def _filter(self, df: DataFrame) -> DataFrame:
        return df.filter(col("images").isNotNull()).filter(col("title").isNotNull()).filter(col("description").isNotNull())

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: dict[str, DataFrame]) -> DataFrame:
        df_books_metadata = df[BooksMetadataTable().table_name]
        df_filtered = self._filter(df_books_metadata)
        return self._only_schema_cols(df_filtered)


@dataclass
class UserReviewModerationTable(Table):
    table_name: str = "user_review_moderation"
    partition_col: str = ""
    parent_tables: tuple[str] = (BooksReviewsTable().table_name,)

    @property
    def schema(self):
        return StructType(
            [
                StructField("user_id", StringType(), False),
                StructField(
                    "reviews",
                    ArrayType(
                        StructType(
                            [
                                StructField("title", StringType(), False),
                                StructField("text", StringType(), False),
                                StructField("timestamp", TimestampType(), False),
                            ]
                        )
                    ),
                    False,
                ),
            ]
        )

    def _filter_cols(self, df: DataFrame) -> DataFrame:
        return df.select("user_id", "title", "text", "timestamp")

    def _aggregate_reviews_to_list(self, df: DataFrame) -> DataFrame:
        windowSpec = Window.partitionBy("user_id").orderBy("timestamp")

        rev_with_row_num = df.withColumn("row_num", F.row_number().over(windowSpec))
        rev_aggregated = rev_with_row_num.groupBy("user_id").agg(F.collect_list(F.struct("title", "text", "timestamp")).alias("reviews"))

        return rev_aggregated

    def _filter(self, df: DataFrame) -> DataFrame:
        return df.filter(col("images").isNotNull()).filter(col("title").isNotNull()).filter(col("description").isNotNull())

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: dict[str, DataFrame]) -> DataFrame:
        df_books_reviews = df[BooksReviewsTable().table_name]
        df_filtered_cols = self._filter_cols(df_books_reviews)
        df_aggregated = self._aggregate_reviews_to_list(df_filtered_cols)
        return self._only_schema_cols(df_aggregated)


@dataclass
class UserFeaturesTable(Table):
    table_name: str = "user_features"
    partition_col: str = ""
    parent_tables: tuple[str] = (BooksReviewsTable().table_name,)

    @property
    def schema(self):
        return StructType(
            [
                StructField("user_id", StringType(), False),
                StructField("n_reviews", LongType(), False),
                StructField("mean_rating", FloatType(), False),
                StructField("mean_helpful_vote", FloatType(), False),
                StructField("std_helpful_vote", FloatType(), False),
            ]
        )

    def _filter(self, df: DataFrame) -> DataFrame:
        cutoff_date = "2023-01-01"
        return df.filter(col("timestamp") < cutoff_date)

    def _aggregate(self, df: DataFrame) -> DataFrame:
        rev_grouped = (
            df.groupBy("user_id")
            .agg(
                F.count("*").alias("n_reviews"),
                F.mean("rating").alias("mean_rating"),
                F.mean("helpful_vote").alias("mean_helpful_vote"),
                F.stddev("helpful_vote").alias("std_helpful_vote"),
            )
            .fillna({"std_helpful_vote": 0})
        )
        return rev_grouped

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: dict[str, DataFrame]) -> DataFrame:
        df_books_reviews = df[BooksReviewsTable().table_name]
        df_books_reviews_filtered = self._filter(df_books_reviews)
        df_books_reviews_aggregated = self._aggregate(df_books_reviews_filtered)
        return self._only_schema_cols(df_books_reviews_aggregated)


@dataclass
class ItemFeaturesTable(Table):
    table_name: str = "item_features"
    partition_col: str = ""
    parent_tables: tuple[str, ...] = (BooksMetadataTable().table_name, BooksReviewsTable().table_name)

    @property
    def schema(self):
        return StructType(
            [
                StructField("parent_asin", StringType(), False),
                StructField("average_rating", FloatType(), False),
                StructField("price", FloatType(), False),
                StructField("main_category", StringType(), False),
                StructField("categories", ArrayType(StringType()), False),
            ]
        )

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: dict[str, DataFrame]) -> DataFrame:
        df_books_metadata = df[BooksMetadataTable().table_name]
        df_books_reviews = df[BooksReviewsTable().table_name]
        return self._only_schema_cols(df_books_metadata)


@dataclass
class ReviewsTable(Table):
    table_name: str = "reviews"
    partition_col: str = ""
    parent_tables: tuple[str] = (BooksReviewsTable().table_name,)

    @property
    def schema(self):
        return StructType(
            [
                StructField("timestamp", TimestampType(), False),
                StructField("user_id", StringType(), False),
                StructField("parent_asin", StringType(), False),
                StructField("rating", FloatType(), False),
            ]
        )

    def _only_schema_cols(self, df: DataFrame) -> DataFrame:
        return df.select(*[col for col in df.columns if col in self.schema.fieldNames()])

    def process(self, df: dict[str, DataFrame]) -> DataFrame:
        df_books_reviews = df[BooksReviewsTable().table_name]
        return self._only_schema_cols(df_books_reviews)


gold_table_mapping = {
    ExplicitContentItemsTable.table_name: ExplicitContentItemsTable(),
    UserReviewModerationTable.table_name: UserReviewModerationTable(),
    UserFeaturesTable.table_name: UserFeaturesTable(),
    ItemFeaturesTable.table_name: ItemFeaturesTable(),
    ReviewsTable.table_name: ReviewsTable(),
}
