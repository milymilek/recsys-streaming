import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv(".env")

# download-data
#DOWNLOAD_DATA_URL = os.getenv('DOWNLOAD_DATA_URL')
DOWNLOAD_DATA_URL="https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz" #"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Handmade_Products.jsonl.gz"
#DOWNLOAD_METADATA_URL = os.getenv('DOWNLOAD_METADATA_URL')
DOWNLOAD_METADATA_URL="https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Subscription_Boxes.jsonl.gz"  # "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Handmade_Products.jsonl.gz"
DATA_DIR = pathlib.Path(".data")
DATA_FILE = DATA_DIR / "ratings"
METADATA_FILE = DATA_DIR / "metadata"

# process-data
DATASET_FILE = DATA_DIR / "dataset"
FEATURE_MAPS_FILE = DATA_DIR / "feature_maps"

# train
RUNS_DIR = pathlib.Path(".runs")

# db
MONGODB_HOST = os.getenv("MONGODB_HOST", default="localhost")
MONGODB_PORT = 27017
MONGODB_AUTHSOURCE = "admin"
MONGODB_USERNAME = "root"
MONGODB_PASSWORD = "root"

# kafka
KAFKA_BROKER_URL = "localhost:9092"
RECOMMENDATIONS_TOPIC = "recommendations"
USER_ACTIONS_TOPIC = "users.actions"