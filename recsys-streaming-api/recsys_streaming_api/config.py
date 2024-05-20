import os

MODEL_URI_PATH = "C:/Users/Milosz/Projects/recsys-streaming/recsys-streaming-ml/.runs/DeepFM/2024-05-11_15-29-20/model.pt"

# db
MONGODB_HOST = os.getenv("MONGODB_HOST", default="localhost")
MONGODB_PORT = 27017
MONGODB_AUTHSOURCE = "admin"
MONGODB_USERNAME = "root"
MONGODB_PASSWORD = "root"