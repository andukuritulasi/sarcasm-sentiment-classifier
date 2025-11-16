from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/deberta_sarcasm"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

label_map = {0: "Not Sarcastic", 1: "Sarcastic"}

def predict_sarcasm(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = float(probs[0][prediction])
    return {
        "label": label_map[prediction],
        "confidence": confidence
    }

if __name__ == "__main__":
    print(predict_sarcasm("Wow great, another delay… absolutely love waiting forever!"))
