from fastapi import FastAPI, Request
from src.predict import Predictor
from hdfs import InsecureClient

app = FastAPI()

predictor = Predictor(params={'mode': 'infere', 'tests': 'none'})

client = InsecureClient('http://127.0.0.1:9870', user='mlops')
client.makedirs('/service_log', permission='766')
client.write('/service_log/log', data='request,class,confidence\n', overwrite=True, permission='766')

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
    
    with client.read('/service_log/log', encoding='utf-8') as reader:
        content = reader.read()

    client.write('/service_log/log', data=content+''.join(log[:]), overwrite=True)

    return {"Result": result}