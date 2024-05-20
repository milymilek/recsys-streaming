import json
import time

import pandas as pd
from kafka import KafkaProducer

from config import RECOMMENDATIONS_TOPIC, BOOTSTRAP_KAFKA_SERVER, MOCK_DATA_PATH

print(RECOMMENDATIONS_TOPIC)
print(BOOTSTRAP_KAFKA_SERVER)


def wait_for_kafka(bootstrap_servers):
    users_actions_producer = None
    while users_actions_producer is None:
        try:
            users_actions_producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
            print("Kafka is ready.")
        except Exception:
            print("Kafka not avaiable, next attemp in 2 sec...")
            time.sleep(2)
    return users_actions_producer


mock_data = pd.read_json(MOCK_DATA_PATH, lines=True)
producer = wait_for_kafka(BOOTSTRAP_KAFKA_SERVER)


def generate_and_send_messages():
    while True:
        message = mock_data.sample()
        message_dict = message.to_dict(orient='records')[0]
        message_json = json.dumps(message_dict).encode('utf-8')
        producer.send(RECOMMENDATIONS_TOPIC, value=message_json)
        producer.flush()
        print(f"Sent message: {message}")
        time.sleep(5)


if __name__ == "__main__":
    generate_and_send_messages()
