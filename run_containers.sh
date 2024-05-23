docker network create --driver bridge recsys-streaming_kafka_network
cd recsys-streaming/kafka
docker compose up 
cd -
docker compose up