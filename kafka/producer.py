from confluent_kafka import Producer
import socket
import json

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))

conf = {'bootstrap.servers': '127.0.0.1:29092',
        'client.id': socket.gethostname()}

producer = Producer(conf)

log_str="test1,test2,test3\ntest4,test5,test6\n"
data = {"text":log_str}
message = json.dumps(data).encode('utf-8')

producer.produce('for-hdfs', key="service-log", value=message, callback=acked)
producer.poll(1)