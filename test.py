import pandas as pd
import json
import pickle
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from train import MultiTaskModel
from safetensors.torch import load_file

MODEL_PATH = "./models/deberta_multitask"
DATA_FILE = "data/ecommerce_conversations.csv"

print("=" * 60)
print("🧪 TESTING DUAL-TASK MODEL (Sarcasm + Sentiment)")
print("=" * 60)

# -------------------------------
# Load Data
# -------------------------------
print("\n📂 Loading test data...")
df = pd.read_csv(DATA_FILE)

# Parse JSON conversations
def parse_conversation(conv_json):
    try:
        conv = json.loads(conv_json)
        return " ".join([turn["text"] for turn in conv])
    except:
        return str(conv_json)

df["text"] = df["conversation_json"].apply(parse_conversation)
df = df.drop_duplicates(subset=["conversation_id"])

# Load encoder
with open(f"{MODEL_PATH}/sentiment_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)

# Encode labels
df["sarcasm"] = df["sarcasm"].astype(int)
df["sentiment_encoded"] = sentiment_encoder.transform(df["sentiment"])

# Split (same split as training - 500 test samples)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=500, random_state=42)

print(f"✓ Total samples: {len(df)}")
print(f"✓ Training samples: {len(train_df)}")
print(f"✓ Test samples: {len(test_df)}")
print(f"✓ Sentiments: {list(sentiment_encoder.classes_)}")

# -------------------------------
# Load Model
# -------------------------------
print("\n🤖 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

num_sentiments = len(sentiment_encoder.classes_)

model = MultiTaskModel(
    "microsoft/deberta-v3-base",
    num_sarcasm_labels=2,
    num_sentiment_labels=num_sentiments
)

state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()
print("✓ Model loaded successfully!")

# -------------------------------
# Make Predictions
# -------------------------------
print(f"\n🔮 Making predictions on {len(test_df)} test samples...")

sarcasm_preds = []
sentiment_preds = []

sarcasm_true = test_df["sarcasm"].tolist()
sentiment_true = test_df["sentiment_encoded"].tolist()

for idx, row in test_df.iterrows():
    text = row["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    sarcasm_pred = torch.argmax(outputs['sarcasm_logits']).item()
    sentiment_pred = torch.argmax(outputs['sentiment_logits']).item()
    
    sarcasm_preds.append(sarcasm_pred)
    sentiment_preds.append(sentiment_pred)

print("✓ Predictions complete!")

# -------------------------------
# Calculate Metrics
# -------------------------------
print("\n" + "=" * 60)
print("📊 EVALUATION RESULTS")
print("=" * 60)

# Sarcasm Metrics
print("\n🎭 SARCASM DETECTION")
print("-" * 60)
sarcasm_acc = accuracy_score(sarcasm_true, sarcasm_preds)
sarcasm_prec, sarcasm_rec, sarcasm_f1, _ = precision_recall_fscore_support(
    sarcasm_true, sarcasm_preds, average='binary'
)
print(f"Accuracy:  {sarcasm_acc:.2%}")
print(f"Precision: {sarcasm_prec:.2%}")
print(f"Recall:    {sarcasm_rec:.2%}")
print(f"F1 Score:  {sarcasm_f1:.2%}")

print("\nClassification Report:")
print(classification_report(sarcasm_true, sarcasm_preds, 
                          target_names=['Not Sarcastic', 'Sarcastic']))

print("\nConfusion Matrix:")
cm = confusion_matrix(sarcasm_true, sarcasm_preds)
print(f"                 Predicted")
print(f"                 Not Sarc  Sarcastic")
print(f"Actual Not Sarc    {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"       Sarcastic   {cm[1][0]:4d}      {cm[1][1]:4d}")

# Sentiment Metrics
print("\n" + "=" * 60)
print("😊 SENTIMENT ANALYSIS")
print("-" * 60)
sentiment_acc = accuracy_score(sentiment_true, sentiment_preds)
sentiment_prec, sentiment_rec, sentiment_f1, _ = precision_recall_fscore_support(
    sentiment_true, sentiment_preds, average='weighted'
)
print(f"Accuracy:  {sentiment_acc:.2%}")
print(f"Precision: {sentiment_prec:.2%}")
print(f"Recall:    {sentiment_rec:.2%}")
print(f"F1 Score:  {sentiment_f1:.2%}")

print("\nClassification Report:")
print(classification_report(sentiment_true, sentiment_preds, 
                          target_names=sentiment_encoder.classes_))

# Overall Summary
print("\n" + "=" * 60)
print("📈 OVERALL SUMMARY")
print("=" * 60)
print(f"Test Samples:        {len(test_df)}")
print(f"Sarcasm Accuracy:    {sarcasm_acc:.2%}")
print(f"Sentiment Accuracy:  {sentiment_acc:.2%}")
print(f"Average Accuracy:    {(sarcasm_acc + sentiment_acc) / 2:.2%}")
print(f"\n💡 Category prediction removed - model now focuses on sarcasm + sentiment!")

# Sample Predictions
print("\n" + "=" * 60)
print("🔍 SAMPLE PREDICTIONS (First 5)")
print("=" * 60)

for i, (idx, row) in enumerate(test_df.head(5).iterrows()):
    print(f"\n--- Sample {i+1} ---")
    print(f"Text: {row['text'][:100]}...")
    print(f"True:      Sarcasm={row['sarcasm']}, Sentiment={row['sentiment']}")
    print(f"Predicted: Sarcasm={sarcasm_preds[i]}, Sentiment={sentiment_encoder.inverse_transform([sentiment_preds[i]])[0]}")
    
    # Check if correct
    correct_sarcasm = "✓" if sarcasm_preds[i] == row['sarcasm'] else "✗"
    correct_sentiment = "✓" if sentiment_preds[i] == row['sentiment_encoded'] else "✗"
    print(f"Correct:   Sarcasm {correct_sarcasm}, Sentiment {correct_sentiment}")

print("\n" + "=" * 60)
print("✅ Testing complete!")
print("=" * 60)
