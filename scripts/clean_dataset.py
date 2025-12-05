"""
Clean existing dataset by removing contradictory labels
Keep only the most common label for duplicate texts
"""
import pandas as pd
import json
from collections import Counter

print("Loading existing dataset...")
df = pd.read_csv("data/ecommerce_conversations_clean.csv")

# Parse conversations to get customer text
def get_customer_text(conv_json):
    try:
        conv = json.loads(conv_json)
        return conv[0]["text"]  # First message is customer
    except:
        return str(conv_json)

df["customer_text"] = df["conversation_json"].apply(get_customer_text)

print(f"Total samples: {len(df)}")

# Find duplicates
duplicates = df[df.duplicated(subset=["customer_text"], keep=False)]
print(f"Duplicate texts found: {len(duplicates['customer_text'].unique())}")

# For each duplicate text, find the most common label
def get_majority_labels(group):
    """Return the most common sarcasm and sentiment labels"""
    sarcasm_counts = Counter(group["sarcasm"])
    sentiment_counts = Counter(group["sentiment"])
    
    majority_sarcasm = sarcasm_counts.most_common(1)[0][0]
    majority_sentiment = sentiment_counts.most_common(1)[0][0]
    
    return majority_sarcasm, majority_sentiment

# Group by customer text and keep one sample per unique text with majority labels
cleaned_samples = []

for text, group in df.groupby("customer_text"):
    majority_sarcasm, majority_sentiment = get_majority_labels(group)
    
    # Take the first sample and update its labels
    sample = group.iloc[0].copy()
    sample["sarcasm"] = majority_sarcasm
    sample["sentiment"] = majority_sentiment
    
    cleaned_samples.append(sample)

# Create cleaned dataframe
df_clean = pd.DataFrame(cleaned_samples)
df_clean = df_clean.drop(columns=["customer_text"])

# Reset conversation IDs
df_clean["conversation_id"] = range(1, len(df_clean) + 1)

# Shuffle
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_file = "data/ecommerce_conversations_deduped.csv"
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Cleaned dataset saved to: {output_file}")
print(f"✅ Samples after deduplication: {len(df_clean)}")
print(f"✅ Removed {len(df) - len(df_clean)} duplicate/contradictory samples")

print(f"\n📊 Distribution:")
print(f"Sarcasm: {df_clean['sarcasm'].value_counts().to_dict()}")
print(f"Sentiment: {df_clean['sentiment'].value_counts().to_dict()}")
print(f"Category: {df_clean['category'].value_counts().to_dict()}")

# Show some examples of what was fixed
print("\n🔍 Examples of deduplicated texts:")
for text, group in list(df.groupby("customer_text"))[:5]:
    if len(group) > 1:
        print(f"\nText: {text[:80]}...")
        print(f"  Original labels: {group[['sarcasm', 'sentiment']].values.tolist()}")
        majority_sarcasm, majority_sentiment = get_majority_labels(group)
        print(f"  Majority label: sarcasm={majority_sarcasm}, sentiment={majority_sentiment}")
