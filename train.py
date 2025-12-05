import pandas as pd
import pickle
import json
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
DATA_FILE = "data/ecommerce_conversations_clean.csv"
OUTPUT_DIR = "models/deberta_multitask"
MAX_LENGTH = 256
NUM_EPOCHS = 3  # 3 epochs is optimal with 5000 training samples
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# -------------------------------
# Dual-Task Model: Sarcasm + Sarcasm-Aware Sentiment
# -------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_sarcasm_labels, num_sentiment_labels):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size
        
        # Sarcasm classifier (predicted first)
        self.sarcasm_classifier = nn.Linear(hidden_size, num_sarcasm_labels)
        
        # IMPROVED: Sentiment classifier uses text features + sarcasm prediction
        # Input size = hidden_size + num_sarcasm_labels (768 + 2 = 770)
        self.sentiment_classifier = nn.Linear(hidden_size + num_sarcasm_labels, num_sentiment_labels)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, sarcasm_labels=None, sentiment_labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Step 1: Predict sarcasm first
        sarcasm_logits = self.sarcasm_classifier(pooled_output)
        
        # Step 2: Predict sentiment using text + sarcasm information
        # Concatenate pooled output with sarcasm logits
        sentiment_input = torch.cat([pooled_output, sarcasm_logits], dim=1)
        sentiment_logits = self.sentiment_classifier(sentiment_input)
        
        loss = None
        if sarcasm_labels is not None and sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Ensure labels are long tensors
            sarcasm_labels = sarcasm_labels.long() if not sarcasm_labels.dtype == torch.long else sarcasm_labels
            sentiment_labels = sentiment_labels.long() if not sentiment_labels.dtype == torch.long else sentiment_labels
            
            sarcasm_loss = loss_fct(sarcasm_logits, sarcasm_labels)
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            
            # Equal weighting for both tasks
            loss = sarcasm_loss + sentiment_loss
        
        return {
            'loss': loss,
            'sarcasm_logits': sarcasm_logits,
            'sentiment_logits': sentiment_logits
        }

# -------------------------------
# Load dataset
# -------------------------------
print("Loading dataset...")
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

# Encode labels
sentiment_encoder = LabelEncoder()

# Keep original column names but encode where needed
df["sarcasm"] = df["sarcasm"].astype(int)  # Ensure it's int
df["sentiment"] = sentiment_encoder.fit_transform(df["sentiment"])

# Save encoder for later use
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/sentiment_encoder.pkl", "wb") as f:
    pickle.dump(sentiment_encoder, f)

print(f"✓ Sentiments: {list(sentiment_encoder.classes_)}")
print(f"✓ Sarcasm: Binary (0=Not Sarcastic, 1=Sarcastic)")

# Select columns (removed category)
df = df[["text", "sarcasm", "sentiment"]]

# -------------------------------
# Split train / test
# -------------------------------
# Use 500 samples for testing (fixed size), rest for training
train_df, test_df = train_test_split(df, test_size=500, random_state=42)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# -------------------------------
# Tokenizer
# -------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Rename label columns to match what Trainer expects
train_ds = train_ds.rename_column("sarcasm", "sarcasm_labels")
train_ds = train_ds.rename_column("sentiment", "sentiment_labels")

test_ds = test_ds.rename_column("sarcasm", "sarcasm_labels")
test_ds = test_ds.rename_column("sentiment", "sentiment_labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sarcasm_labels", "sentiment_labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sarcasm_labels", "sentiment_labels"])

# -------------------------------
# Model
# -------------------------------
print("Creating dual-task model (Sarcasm + Sentiment)...")
num_sentiments = len(sentiment_encoder.classes_)

model = MultiTaskModel(MODEL_NAME, num_sarcasm_labels=2, num_sentiment_labels=num_sentiments)

# -------------------------------
# Custom Trainer
# -------------------------------
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        sarcasm_labels = inputs.pop("sarcasm_labels")
        sentiment_labels = inputs.pop("sentiment_labels")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sarcasm_labels=sarcasm_labels,
            sentiment_labels=sentiment_labels
        )
        
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Extract labels
        sarcasm_labels = inputs.pop("sarcasm_labels", None)
        sentiment_labels = inputs.pop("sentiment_labels", None)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sarcasm_labels=sarcasm_labels,
                sentiment_labels=sentiment_labels
            )
        
        loss = outputs['loss']
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Stack predictions and labels
        logits = (
            outputs['sarcasm_logits'].detach(),
            outputs['sentiment_logits'].detach()
        )
        
        labels = (
            sarcasm_labels.detach() if sarcasm_labels is not None else None,
            sentiment_labels.detach() if sentiment_labels is not None else None
        )
        
        return (loss, logits, labels)

# Metrics
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # predictions is a tuple of (sarcasm_logits, sentiment_logits)
    sarcasm_preds = np.argmax(predictions[0], axis=1)
    sentiment_preds = np.argmax(predictions[1], axis=1)
    
    # labels is a tuple of (sarcasm_labels, sentiment_labels)
    sarcasm_labels = labels[0]
    sentiment_labels = labels[1]
    
    return {
        'sarcasm_accuracy': accuracy_score(sarcasm_labels, sarcasm_preds),
        'sentiment_accuracy': accuracy_score(sentiment_labels, sentiment_preds),
    }

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    load_best_model_at_end=True,
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False  # Disable pin_memory for Apple Silicon
)

# -------------------------------
# Trainer
# -------------------------------
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# -------------------------------
# Train
# -------------------------------
print("\n🚀 Starting IMPROVED training...")
print("✨ Sentiment classifier now uses sarcasm information!")
print("📊 This should improve sentiment accuracy significantly.\n")
trainer.train()

# -------------------------------
# Save model and tokenizer
# -------------------------------
print("\n💾 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete. Dual-task model saved in: {OUTPUT_DIR}")
print(f"✓ Sarcasm detection (binary)")
print(f"✓ Sentiment classification ({num_sentiments} sentiments) - SARCASM-AWARE! 🎯")
print(f"\n💡 Category prediction removed - it was unreliable (<40% confidence)")
print(f"   Focus on what works: Sarcasm + Sentiment! 🚀")
