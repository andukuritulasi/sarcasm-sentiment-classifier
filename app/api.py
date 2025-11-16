from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_sarcasm

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    return predict_sarcasm(input.text)

