# Check if the network exists
$networkExists = docker network ls | Select-String -Pattern "recsys-streaming_kafka_network"

if (-Not $networkExists) {
    Write-Host "Creating network 'recsys-streaming_kafka_network'..."
    docker network create --driver bridge recsys-streaming_kafka_network
} else {
    Write-Host "Network 'recsys-streaming_kafka_network' already exists. Proceeding..."
}

# Run kafka-related containers
make build_kafka

# Run spark and databases
make build_spark

# Download data in spark container from remote host passed as env
make download_data

# Insert downloaded data into MongoDB
make insert_in_db

# Preprocess data for training
make process_data

# Train first model on basic data and save it to MongoDB
make train

# For streaming run `make stream_recommendations` or `make stream_retrain`...
