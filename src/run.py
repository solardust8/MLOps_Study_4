from fastapi import FastAPI, Request
from src.predict import Predictor
import os
import socket
from confluent_kafka import Producer
import json

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))

conf = {'bootstrap.servers': '127.0.0.1:29092',
        'client.id': socket.gethostname()}

producer = Producer(conf)

predictor = Predictor(params={'mode': 'infere', 'tests': 'none'})
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is Spam/Ham message type prediction model service. Please, use another method to pass prompts in order to get results."}

@app.get("/predict")
async def predict_msg_type(request: Request):
    prompts = await request.json()
    result = predictor.get_model_output(prompts=prompts['text'])

    log = []
    for req, resp in zip(prompts['text'], result):
        log_str = '"'+req+'"'+','+resp['label']+','+str(resp['score'])+'\n'
        log.append(log_str)
    
    #======= SEND NESSAGE TO KAFKA
    data = {"text":''.join(log[:])}
    message = json.dumps(data).encode('utf-8')
    producer.produce('for-hdfs', key="service-log", value=message, callback=acked)
    producer.poll(1)

    return {"Result": result}