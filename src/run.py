from fastapi import FastAPI, Request
from src.predict import Predictor
#from pydantic import BaseModel

app = FastAPI()

predictor = Predictor(params={'mode': 'infere', 'tests': 'none'})

@app.get("/")
async def root():
    return {"message": "This is Spam/Ham message type prediction model service. Please, use another method to pass prompts in order to get results."}

@app.get("/predict")
async def predict_msg_type(request: Request):
    prompts = await request.json()
    result = predictor.get_model_output(prompts=prompts['text'])
    return {"Result": result}