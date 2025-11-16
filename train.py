import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

MODEL_NAME = "microsoft/deberta-v3-base"

# Load dataset
df = pd.read_csv("data/ecommerce_conversations_2500.csv")

# Merge multi-turn conversation for training
df["text"] = df.groupby("conversation_id")["message"].transform(lambda x: " ".join(x))
df = df.drop_duplicates(subset=["conversation_id"])
df = df[["text", "sarcasm"]]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir="models/deberta_sarcasm",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

trainer.save_model("models/deberta_sarcasm")
tokenizer.save_pretrained("models/deberta_sarcasm")

print("Training complete. Model saved!")
