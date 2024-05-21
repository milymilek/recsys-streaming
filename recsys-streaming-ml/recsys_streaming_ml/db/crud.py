from typing import Any

import pandas as pd
from pymongo.database import Database


def insert_df_to_mongo(db: Database, df: pd.DataFrame, collection: str) -> None:
    records = df.to_dict(orient='records')

    db[collection].insert_many(records)


def read_df_from_mongo(db: Database, collection: str) -> pd.DataFrame:
    cursor = db[collection].find()
    
    # Convert the cursor to a list of dictionaries
    records = list(cursor)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(records)
    
    # Optionally, remove the MongoDB ObjectId field if it's not needed
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    
    return df


def read_latest_model_version_document(db: Database, collection: str = "model_versions", timestamp_col: str = "timestamp") -> dict[str, Any]:
    latest_version = db[collection].find_one({}, sort=[(timestamp_col, -1)])

    return latest_version