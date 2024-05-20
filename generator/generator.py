import time
import random
import json
import uuid
import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

#KAFKA_SERVER_URI = 'kafka:9092'
KAFKA_SERVER_URI = 'localhost:29092'
TOPIC = 'user_activity'


def _get_kafka_producer():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_SERVER_URI],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            return producer
        except NoBrokersAvailable:
            print("No brokers available, retrying in 5 seconds...")
            time.sleep(5)

def _time_spent():
    crnt_time_seconds = datetime.datetime.now().second
    return abs(5 * (((crnt_time_seconds-1) // 5) % 2) - ((crnt_time_seconds-1) % 5))


def _generate_user_activity(producer: KafkaProducer):
    while True:
        data = {"user_id": str(uuid.uuid4())}
        producer.send(TOPIC, data)
        print(f"Sent: f{data}")
        time.sleep(1)


def main():
    producer = producer = KafkaProducer(
                bootstrap_servers=[KAFKA_SERVER_URI],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

    _generate_user_activity(producer=producer)
    

if __name__ == "__main__":
    main()