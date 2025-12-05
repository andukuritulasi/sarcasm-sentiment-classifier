from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append('..')
from inference import predict_all

app = FastAPI(title="Sarcasm-Sentiment Analyzer API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    """Predict sarcasm and sentiment for given text"""
    return predict_all(input.text)

