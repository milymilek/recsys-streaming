from pymongo import MongoClient
from pymongo.errors import OperationFailure

from recsys_streaming_api.config import MONGODB_HOST, MONGODB_PORT, MONGODB_AUTHSOURCE, MONGODB_USERNAME, MONGODB_PASSWORD

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

client = mongo_client()

#client['model_versions'].insert_many([{"model_name": "deepfm1", "file": b"123456"}])
#print(list(client['model_versions'].aggregate([{"$match": {}}])))