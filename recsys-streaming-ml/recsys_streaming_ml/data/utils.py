import pickle
import pathlib

from recsys_streaming_ml.config import FEATURE_MAPS_FILE

def dump_feature_maps(**kwargs):
    with open(FEATURE_MAPS_FILE.with_suffix(".pkl"), 'wb') as f:
        pickle.dump(kwargs, f)


def load_feature_maps(path: pathlib.Path = FEATURE_MAPS_FILE.with_suffix(".pkl")):
    with open(path, 'rb') as f:
        return pickle.load(f)