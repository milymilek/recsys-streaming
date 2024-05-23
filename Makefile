build_spark:
	docker compose -f ./docker-compose.yaml up -d

build_kafka:
	docker compose -f ./recsys-streaming/docker-compose.yaml up -d

download_data:
	docker exec -it recsys-streaming-ml python -m run --script download_data

insert_in_db:
	docker exec -it recsys-streaming-ml python -m run --script insert_in_db

process_data:
	docker exec -it recsys-streaming-ml python -m run --script process_data

train:
	docker exec -it recsys-streaming-ml python -m run --script train

stream_recommendations:
	docker exec -it recsys-streaming-ml python ./recsys_streaming_ml/spark/stream_recommendations.py

stream_retrain:
	docker exec -it recsys-streaming-ml python ./recsys_streaming_ml/spark/stream_users_actions.py