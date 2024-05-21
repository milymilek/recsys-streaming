import pickle
import pathlib

from recsys_streaming_ml.config import FEATURE_MAPS_FILE, FEATURE_STORE_FILE
from recsys_streaming_ml.db import read_df_from_mongo

def dump_feature_maps(**kwargs):
    with open(FEATURE_MAPS_FILE.with_suffix(".pkl"), 'wb') as f:
        pickle.dump(kwargs, f)


def load_feature_maps(path: pathlib.Path = FEATURE_MAPS_FILE.with_suffix(".pkl")) -> dict[str, int]:
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def build_reverse_feature_maps(feature_maps) -> dict[int, str]:
    rev_fm = {}
    for k in feature_maps.keys():
        rev_fm[k] = {v:k for k,v in feature_maps[k].items()}

    return rev_fm
    

def read_item_feature_store(db, feature_maps, collection='metadata'):
    item_feature_store_raw = read_df_from_mongo(db=db, collection=collection)
    item_feature_store = item_feature_store_raw.copy()
    item_feature_store['parent_asin'] = item_feature_store['parent_asin'].map(feature_maps['parent_id_map'])
    item_feature_store['store_id'] = item_feature_store['store'].map(feature_maps['store_id_map'])
    item_feature_store = item_feature_store.drop(columns='store').dropna().astype(int).sort_values(by='parent_asin').reset_index(drop=True)

    return item_feature_store