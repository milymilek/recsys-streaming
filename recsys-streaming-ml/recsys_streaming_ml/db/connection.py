import json
import time
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from pymongo import MongoClient
from pymongo.errors import OperationFailure, CollectionInvalid

from recsys_streaming_ml.config import MONGODB_HOST, MONGODB_PORT, MONGODB_AUTHSOURCE, MONGODB_USERNAME, MONGODB_PASSWORD

def mongo_client(*args, **kwargs):
    try:
        client = MongoClient(
            host=MONGODB_HOST,
            port=MONGODB_PORT,
            authSource=MONGODB_AUTHSOURCE,
            username=MONGODB_USERNAME,
            password=MONGODB_PASSWORD
        )
        db = client.admin
        client.server_info()
        db.create_collection("model_versions", check_exists=False)

        print("[MONGO] Connection successful")
        return db

    except OperationFailure as e:
        print("[MONGO] Connection failed:", e)
    except Exception:
        print("[MONGO] Connection failed")

mongo_db = mongo_client()
