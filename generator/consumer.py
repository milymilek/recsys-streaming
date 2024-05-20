import json
from kafka import KafkaConsumer

KAFKA_SERVER_URI = 'localhost:9092'
TOPIC = 'recommendations'

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_SERVER_URI],
        auto_offset_reset='earliest',
    )
    
    for message in consumer:
        data = json.loads(message.value)
        print(f"Received: {data}")

if __name__ == "__main__":
    main()
