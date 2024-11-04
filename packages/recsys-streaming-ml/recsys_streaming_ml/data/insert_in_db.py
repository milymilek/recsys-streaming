import pandas as pd
import json
import pathlib

from recsys_streaming_ml.config import (
    MODEL_COLS, META_MODEL_COLS, DATA_FILE, METADATA_FILE
)
from recsys_streaming_ml.db import mongo_db, insert_df_to_mongo


def _read_jsonl(file: pathlib.Path):
    with open(file, 'r') as fp:
        dct = [json.loads(line.strip()) for line in fp]
        return pd.DataFrame(dct)


def _load_data(path: pathlib.Path) -> pd.DataFrame:
    df = _read_jsonl(path.with_suffix(".jsonl"))
    return df


def _filter_cols(df, cols):
    return df[cols]


def run():
    print("SCRIPT: Insert in DB - START")

    names_list = ["ratings", "metadata"]
    datafiles_list = [DATA_FILE, METADATA_FILE]
    cols_list = [MODEL_COLS, META_MODEL_COLS]

    for name, datafile, cols in zip(names_list, datafiles_list, cols_list):
        df = _load_data(path=datafile)
        df = _filter_cols(df, cols)
        insert_df_to_mongo(db=mongo_db, df=df, collection=name)

    print(f'Data inserted to MongoDB.')
    print("SCRIPT: Insert in DB - END")
