from transformers import AutoTokenizer
import torch
import pickle
import os
from safetensors.torch import load_file

MODEL_PATH = "./models/deberta_multitask"

# Load sentiment encoder
with open(f"{MODEL_PATH}/sentiment_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)

num_sentiments = len(sentiment_encoder.classes_)

print(f"Loading dual-task model (Sarcasm + Sentiment)...")
print(f"Sentiments: {list(sentiment_encoder.classes_)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# Load model
from train import MultiTaskModel
model = MultiTaskModel(
    "microsoft/deberta-v3-base", 
    num_sarcasm_labels=2, 
    num_sentiment_labels=num_sentiments
)

# Load weights from safetensors
state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")

sarcasm_map = {0: "Not Sarcastic", 1: "Sarcastic"}

def predict_all(text: str):
    """Predict sarcasm and sentiment for given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Get predictions
    sarcasm_probs = torch.softmax(outputs['sarcasm_logits'], dim=1)
    sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1)
    
    sarcasm_pred = torch.argmax(sarcasm_probs).item()
    sentiment_pred = torch.argmax(sentiment_probs).item()
    
    return {
        "sarcasm": {
            "label": sarcasm_map[sarcasm_pred],
            "confidence": float(sarcasm_probs[0][sarcasm_pred])
        },
        "sentiment": {
            "label": sentiment_encoder.inverse_transform([sentiment_pred])[0],
            "confidence": float(sentiment_probs[0][sentiment_pred])
        }
    }

if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Wow great, another delay… absolutely love waiting forever!",  # Sarcastic
        "The product arrived on time and works perfectly. Thanks!",    # Not sarcastic
    ]
    
    for test_text in test_texts:
        result = predict_all(test_text)
        print(f"\nText: {test_text}")
        print(f"Predictions:")
        print(f"  Sarcasm: {result['sarcasm']['label']} ({result['sarcasm']['confidence']:.2%})")
        print(f"  Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.2%})")
        print("-" * 80)
