import pandas as pd
import json
import pathlib

from recsys_streaming_ml.config import (
    DATA_FILE, TIMESTAMP_COL
)


def _read_jsonl(file: pathlib.Path):
    with open(file, 'r') as fp:
        dct = [json.loads(line.strip()) for line in fp]
        return pd.DataFrame(dct)


def _load_data(path: pathlib.Path) -> pd.DataFrame:
    df = _read_jsonl(path.with_suffix(".jsonl"))
    return df


def _sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=TIMESTAMP_COL)


def _extract_data(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac_index = int(df.shape[0] * frac)

    return df[:frac_index], df[frac_index:]


def _save_to_jsonl(df: pd.DataFrame, filename):
    df.to_json(filename, orient='records', lines=True)


def run():
    print("SCRIPT: Extract presentation data - START")

    names_list = ["ratings"]
    datafiles_list = [DATA_FILE]

    for name, datafile in zip(names_list, datafiles_list):
        df = _load_data(path=datafile)
        df = _sort_data(df)
        df, df_presentation = _extract_data(df, frac=0.8)
        _save_to_jsonl(df, datafile.with_suffix(".jsonl"))
        _save_to_jsonl(df_presentation, (datafile.parent / "presentation").with_suffix(".jsonl"))

    print(f'Presentation data extracted.')
    print("SCRIPT: Extract presentation data - END")
