import pandas as pd
import pathlib
import json
import pickle
from sklearn.model_selection import TimeSeriesSplit

from recsys_streaming_ml.config import DATA_DIR, DATA_FILE, METADATA_FILE, DATASET_FILE
from recsys_streaming_ml.data.utils import dump_feature_maps

MODEL_COLS = ['timestamp', 'rating', 'user_id', 'parent_asin']
META_MODEL_COLS = ['parent_asin', 'store']


def _read_jsonl(file: pathlib.Path):
    with open(file, 'r') as fp:
        dct = [json.loads(line.strip()) for line in fp]
        return pd.DataFrame(dct)


def _load_data(path_ratings: pathlib.Path, path_metadata: pathlib.Path) -> pd.DataFrame:
    df_ratings = _read_jsonl(path_ratings.with_suffix(".jsonl"))
    df_metadata = _read_jsonl(path_metadata.with_suffix(".jsonl"))
    return df_ratings, df_metadata


def _build_map(df, col):
    unique_vals = df[col].unique()
    return dict(zip(unique_vals, range(unique_vals.size)))


def _filter_low_occurence(df, col, threshold=5):
    occ = df[col].value_counts()
    occ = occ[(occ >= threshold)].index
    print(occ)
    return df[df[col].isin(occ)]


def _preprocess_data(df_ratings: pd.DataFrame, df_metadata: pd.DataFrame) -> pd.DataFrame:
    df_ratings = df_ratings[MODEL_COLS]
    df_metadata = df_metadata[META_MODEL_COLS]

    df = df_ratings.sort_values(by='timestamp')
    df = df.merge(df_metadata, left_on='parent_asin', right_on='parent_asin')
    df = df.dropna(subset=['store'])

    USER_ID_MAP = _build_map(df, 'user_id')
    PARENT_ASIN_MAP = _build_map(df, 'parent_asin')
    STORE_MAP = _build_map(df, 'store')
    dump_feature_maps(user_id_map=USER_ID_MAP, parent_id_map=PARENT_ASIN_MAP, store_id_map=STORE_MAP)

    df['user_id'] = df['user_id'].map(USER_ID_MAP)
    df['parent_asin'] = df['parent_asin'].map(PARENT_ASIN_MAP)
    df['store'] = df['store'].map(STORE_MAP)

    return df




def _split_data(df: pd.DataFrame, col: str, validation_ratio: float, test_ratio: float) -> dict[str, pd.DataFrame]:
    df_sorted = df.sort_values(by=col)
    split_training = 1.0 - validation_ratio - test_ratio
    assert split_training >= 0.5, "training split must be at least 50% of all data"
    training_split_point = int(df_sorted.shape[0] * split_training)
    validation_split_point = int(df_sorted.shape[0] * (split_training + validation_ratio))

    df_train = df_sorted[:training_split_point]
    df_validation = df_sorted[training_split_point:validation_split_point]
    df_test = df_sorted[validation_split_point:]

    return {
        "train_data": df_train.drop(columns=['rating', 'timestamp'], axis=1),
        "train_targets": df_train[['rating']],
        "valid_data": df_validation.drop(columns=['rating', 'timestamp'], axis=1),
        "valid_targets": df_validation[['rating']],
        "test_data": df_test.drop(columns=['rating', 'timestamp'], axis=1),
        "test_targets": df_test[['rating']]
    }


def _dump_data(data: dict, path: pathlib.Path) -> None:
    with open(path.with_suffix(".pkl"), 'wb') as f:
        pickle.dump(data, f)


def run():
    print("SCRIPT: Process data - START")

    df_ratings, df_metadata = _load_data(path_ratings=DATA_FILE, path_metadata=METADATA_FILE)
    filtered_data = _preprocess_data(df_ratings, df_metadata)
    splitted_data = _split_data(filtered_data, col='timestamp', validation_ratio=0.2, test_ratio=0.1)
    _dump_data(splitted_data, path=DATASET_FILE)

    print(f'')
    print("SCRIPT: Process data - END")