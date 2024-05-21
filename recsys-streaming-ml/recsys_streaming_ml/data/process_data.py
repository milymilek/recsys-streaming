import pandas as pd
import pathlib
import json
import pickle
from sklearn.model_selection import TimeSeriesSplit

import pyspark.pandas as ps
from pyspark.sql.functions import rand

from recsys_streaming_ml.config import DATASET_FILE, TIMESTAMP_COL
from recsys_streaming_ml.data.utils import dump_feature_maps
from recsys_streaming_ml.db import mongo_db, read_df_from_mongo
from recsys_streaming_ml.spark.utils import spark



def _build_map(df, col):
    unique_vals = df[col].unique().to_numpy()
    return dict(zip(unique_vals, range(len(unique_vals))))


def _filter_low_occurence(df, col, threshold=5):
    occ = df[col].value_counts()
    occ = occ[(occ >= threshold)].index
    print(occ)
    return df[df[col].isin(occ)]


def _preprocess_data(df_ratings: pd.DataFrame, df_metadata: pd.DataFrame) -> pd.DataFrame:
    #df = df_ratings.sort_values(by=TIMESTAMP_COL) # Dont sort by time as we want to include streamed data in training
    df = df_ratings.merge(df_metadata, left_on='parent_asin', right_on='parent_asin')
    df = df.dropna(subset=['store'])

    USER_ID_MAP = _build_map(df, 'user_id')
    PARENT_ASIN_MAP = _build_map(df, 'parent_asin')
    STORE_MAP = _build_map(df, 'store')
    dump_feature_maps(user_id_map=USER_ID_MAP, parent_id_map=PARENT_ASIN_MAP, store_id_map=STORE_MAP)

    df['user_id'] = df['user_id'].map(USER_ID_MAP)
    df['parent_asin'] = df['parent_asin'].map(PARENT_ASIN_MAP)
    df['store'] = df['store'].map(STORE_MAP)

    return df




def _split_data(df: pd.DataFrame, validation_ratio: float, test_ratio: float) -> dict[str, pd.DataFrame]:
    #df = df_ratings.sort_values(by=TIMESTAMP_COL) # Dont sort by time as we want to include streamed data in training

    # Shuffle df
    #df = df.to_spark().select("*").orderBy(rand()).pandas_api()

    split_training = 1.0 - validation_ratio - test_ratio
    assert split_training >= 0.5, "training split must be at least 50% of all data"
    training_split_point = int(df.shape[0] * split_training)
    validation_split_point = int(df.shape[0] * (split_training + validation_ratio))

    df_train = df[:training_split_point]
    df_validation = df[training_split_point:validation_split_point]
    df_test = df[validation_split_point:]

    return {
        "train_data": df_train.drop(columns=['rating', 'timestamp'], axis=1),
        "train_targets": df_train[['rating']],
        "valid_data": df_validation.drop(columns=['rating', 'timestamp'], axis=1),
        "valid_targets": df_validation[['rating']],
    }


def _dump_data(data: dict, path: pathlib.Path) -> None:
    for k, df in data.items():
        df.to_pandas().to_csv((path / f"{k}.csv").as_posix(), index=False)
        #df.to_spark().write.format("csv").mode('overwrite').save(path.as_posix())

    # with open(path.with_suffix(".pkl"), 'wb') as f:
    #     pickle.dump(data, f)


def run():
    print("SCRIPT: Process data - START")
    session = spark()

    _names_list = ["ratings", "metadata"]

    df_ratings_stream = pd.DataFrame() #read_stream(...)
    df_ratings_historical = read_df_from_mongo(db=mongo_db, collection=_names_list[0])
    df_ratings = pd.concat([df_ratings_stream, df_ratings_historical])
    df_metadata = read_df_from_mongo(db=mongo_db, collection=_names_list[1])

    psdf_ratings = ps.from_pandas(df_ratings)
    psdf_metadata = ps.from_pandas(df_metadata)

    filtered_data = _preprocess_data(psdf_ratings, psdf_metadata)
    splitted_data = _split_data(filtered_data, validation_ratio=0.2, test_ratio=0.0)
    _dump_data(splitted_data, path=DATASET_FILE)

    print(f'')
    print("SCRIPT: Process data - END")