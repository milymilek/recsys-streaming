import json
from kafka import KafkaConsumer

KAFKA_SERVER_URI = 'kafka:9092'
TOPIC = 'user_activity'

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_SERVER_URI],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group'
    )
    
    for message in consumer:
        data = json.loads(message.value)
        print(f"Received: {data}")

if __name__ == "__main__":
    main()
