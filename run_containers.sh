#/bin/bash
docker network create app_network
cd recsys-streaming/kafka
docker compose up 
cd -
docker compose up