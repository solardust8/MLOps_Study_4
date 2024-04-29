from confluent_kafka import Consumer, KafkaException, KafkaError
import threading
import sys 
import json
from hdfs import InsecureClient
import hvac
import os


running = True

def msg_process(message, client_handler):
    payload=message.value().decode('utf-8')
    data=json.loads(payload)
    #print(data['text'])
    with client_handler.read('/service_log/log', encoding='utf-8') as reader:
        content = reader.read()
    client_handler.write('/service_log/log', data=content+data['text'], overwrite=True)


def basic_consume_loop(consumer, topics, client_handler):
    try:
        consumer.subscribe(topics)

        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None: continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                msg_process(msg, client_handler)
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()

def shutdown():
    global running
    running = False

if __name__=='__main__':

    # ========= INIT KAFKA CONSUMER
    conf = {'bootstrap.servers': '127.0.0.1:29092',
        'group.id': 'local',
        'auto.offset.reset': 'smallest'}

    consumer = Consumer(conf)

    #========== INIT VAULT CLIENT 
    hvac_token = os.getenv('HVAC_TOKEN')

    hvac_client = hvac.Client(
        url='http://127.0.0.1:8200',
        token=hvac_token
    )

    #========== GET SECRETS

    read_response = hvac_client.secrets.kv.read_secret_version(path='hdfs-secret')
    hdfs_secret_part1 = read_response['data']['data']['part1']
    hdfs_secret_part2 = read_response['data']['data']['part2']

    #========== INIT HDFS CLIENT
    hdfs_client = InsecureClient(f'http://{hdfs_secret_part1}:{hdfs_secret_part2}', user='mlops')
    hdfs_client.makedirs('/service_log', permission='766')
    hdfs_client.write('/service_log/log', data='request,class,confidence\n', overwrite=True, permission='766')



    consumer_thread = threading.Thread(target=basic_consume_loop(consumer, ['for-hdfs'], hdfs_client))
    consumer_thread.start()
    

