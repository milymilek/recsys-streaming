import os
import json
import time
import pandas as pd
from kafka import KafkaProducer


TOPIC = os.getenv("TOPIC", None)
BOOTSTRAP_KAFKA_SERVER = os.getenv("BOOTSTRAP_KAFKA_SERVER", "kafka0:9093")
MOCK_DATA_PATH = os.getenv("MOCK_DATA_PATH", None)
SEND_INTERVAL = os.getenv("SEND_INTERVAL", 5)

print(f'Config: \n{TOPIC=}\n{BOOTSTRAP_KAFKA_SERVER=}\n{MOCK_DATA_PATH=}\n{SEND_INTERVAL=}\n')


def _wait_for_kafka(bootstrap_servers: str) -> KafkaProducer:
    users_actions_producer = None
    while users_actions_producer is None:
        try:
            users_actions_producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
            print("Kafka is ready.")
        except Exception:
            print("Kafka not avaiable, next attemp in 2 sec...")
            time.sleep(2)
    return users_actions_producer


def _generate_and_send_messages(producer: KafkaProducer, data: pd.DataFrame, topic: str, interval: float) -> None:
    while True:
        message = data.sample()
        message_dict = message.to_dict(orient='records')[0]
        message_json = json.dumps(message_dict).encode('utf-8')
        producer.send(topic, value=message_json)
        producer.flush()
        print(f"Sent message: {message}")
        time.sleep(interval)


def main() -> None:
    mock_data = pd.read_json(MOCK_DATA_PATH, lines=True)
    producer = _wait_for_kafka(bootstrap_servers=BOOTSTRAP_KAFKA_SERVER)

    _generate_and_send_messages(producer=producer, data=mock_data, topic=TOPIC, interval=float(SEND_INTERVAL))


if __name__ == "__main__":
    main()
