from fastapi import FastAPI, Request
from src.predict import Predictor
from hdfs import InsecureClient
import hvac
import os

hvac_token = os.getenv('HVAC_TOKEN')

hvac_client = hvac.Client(
    url='http://127.0.0.1:8200',
    token=hvac_token
)

read_response = hvac_client.secrets.kv.read_secret_version(path='hdfs-secret')
hdfs_secret_part1 = read_response['data']['data']['part1']
hdfs_secret_part2 = read_response['data']['data']['part2']

hdfs_client = InsecureClient(f'http://{hdfs_secret_part1}:{hdfs_secret_part2}', user='mlops')
hdfs_client.makedirs('/service_log', permission='766')
hdfs_client.write('/service_log/log', data='request,class,confidence\n', overwrite=True, permission='766')

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
    
    with hdfs_client.read('/service_log/log', encoding='utf-8') as reader:
        content = reader.read()

    hdfs_client.write('/service_log/log', data=content+''.join(log[:]), overwrite=True)

    return {"Result": result}