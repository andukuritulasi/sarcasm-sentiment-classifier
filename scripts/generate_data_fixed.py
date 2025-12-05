"""
Generate high-quality e-commerce conversation samples with CONSISTENT labeling
"""
import pandas as pd
import json
import random

# Templates with EXPLICIT sentiment mapping
sarcastic_negative_templates = [
    ("Oh great, {issue}. Just what I needed today.", "I apologize for the inconvenience. Let me help you with that."),
    ("Wow, {issue}. You guys are really on top of things.", "I understand your frustration. Let me look into this."),
    ("Perfect timing! {issue}. Couldn't be better.", "I'm sorry to hear that. Let me check your account."),
    ("Fantastic! {issue}. This is exactly what I paid for.", "I sincerely apologize. Let me resolve this for you."),
    ("Oh wonderful, {issue}. Best service ever.", "I'm very sorry. Let me investigate this immediately."),
    ("Amazing! {issue}. You never disappoint.", "I understand your concern. Please give me a moment."),
    ("Brilliant! {issue}. Keep up the great work.", "I apologize for this issue. Let me help you right away."),
    ("Lovely! {issue}. This is why I shop here.", "I'm sorry for the trouble. Let me fix this for you."),
    ("Superb! {issue}. Really impressed with your service.", "I apologize. Let me check what went wrong."),
    ("Excellent! {issue}. You've outdone yourselves.", "I'm sorry to hear this. Let me assist you immediately."),
    ("Sure, {issue}. I absolutely love this.", "I understand your concern. Please allow me a moment."),
]

# Non-sarcastic negative (actual problems)
problem_templates = [
    ("I have a problem with {issue}. Can you help?", "Of course! Let me look into that for you."),
    ("There's an issue with {issue}. Please assist.", "I'd be happy to help. Let me check your account."),
    ("{issue}. What can be done about this?", "Let me investigate this for you right away."),
    ("I need help with {issue}.", "I'll help you resolve this. One moment please."),
    ("Can you help me with {issue}?", "Absolutely! Let me check that for you."),
    ("I'm experiencing {issue}. Please help.", "I'm here to help. Let me look into this."),
    ("{issue}. Could you please fix this?", "Of course. Let me resolve this for you."),
    ("I'm having trouble with {issue}.", "I understand. Let me assist you with that."),
    ("There seems to be {issue}. Can you check?", "Certainly. Let me investigate this for you."),
    ("{issue}. I need this resolved please.", "I'll help you with that right away."),
]

# Non-sarcastic positive
positive_templates = [
    ("Thank you for {action}. Great service!", "You're welcome! We're glad we could help."),
    ("I appreciate {action}. Very helpful.", "We're happy to assist! Is there anything else?"),
    ("{action} was perfect. Thanks!", "Glad we could help! Have a great day."),
    ("Excellent service with {action}. Thank you!", "Thank you for your feedback! We appreciate it."),
    ("Really happy with {action}. Well done!", "We're thrilled to hear that! Thank you."),
    ("Great job on {action}. Much appreciated.", "Thank you! We're here if you need anything else."),
    ("{action} exceeded my expectations. Thanks!", "We're so glad! Thank you for choosing us."),
    ("Very satisfied with {action}. Thank you.", "We appreciate your business! Have a wonderful day."),
    ("The {action} was handled perfectly. Thanks!", "Thank you for the kind words! We're here to help."),
    ("Impressed with {action}. Keep it up!", "Thank you! We strive to provide the best service."),
]

# Neutral queries (no strong sentiment)
neutral_templates = [
    ("How can I track my shipment?", "Let me review your order details."),
    ("What's the status of my order?", "I'll check that for you right away."),
    ("Can you tell me about your return policy?", "Of course! Let me explain that for you."),
    ("I'd like to update my shipping address.", "I can help you with that."),
    ("When will my order arrive?", "Let me check the delivery estimate for you."),
    ("Do you have this in a different size?", "Let me check our inventory for you."),
    ("Can I change my order?", "Let me see what I can do for you."),
    ("What payment methods do you accept?", "I'd be happy to explain our payment options."),
]

# Issue types (negative context)
negative_issues = [
    "my order hasn't arrived",
    "the wrong item was delivered",
    "I was charged twice",
    "my package is damaged",
    "the product doesn't work",
    "my refund hasn't been processed",
    "the item is defective",
    "I received an empty box",
    "the delivery address is wrong",
    "my account is locked",
    "the discount code doesn't work",
    "the item is out of stock",
    "my payment failed",
    "the product description was misleading",
    "the size is incorrect",
    "my order was cancelled without notice",
    "the shipping cost is too high",
    "the product arrived broken",
    "my order was delayed again",
]

# Positive actions
positive_actions = [
    "resolving my issue",
    "the quick response",
    "processing my refund",
    "the fast delivery",
    "helping me track my order",
    "the replacement item",
    "fixing the billing error",
    "the customer support",
    "handling my complaint",
    "the product quality",
    "updating my order",
    "the clear communication",
    "resolving this quickly",
    "the professional service",
    "helping me understand",
]

categories = ["delivery_issue", "technical_support", "billing", "product_quality", "general_query"]

# Generate samples with CONSISTENT labeling
samples = []
conversation_id = 1

print("Generating 5500 high-quality samples with consistent labels...")

# 1. Sarcastic + Negative (1500 samples)
for i in range(1500):
    template = random.choice(sarcastic_negative_templates)
    issue = random.choice(negative_issues)
    customer_text = template[0].format(issue=issue)
    agent_text = template[1]
    
    conversation = [
        {"speaker": "customer", "text": customer_text},
        {"speaker": "agent", "text": agent_text}
    ]
    
    samples.append({
        "conversation_id": conversation_id,
        "conversation_json": json.dumps(conversation),
        "sarcasm": 1,
        "category": random.choice(categories),
        "sentiment": "negative"  # Sarcasm about problems = negative
    })
    conversation_id += 1

# 2. Non-sarcastic + Negative (1500 samples)
for i in range(1500):
    template = random.choice(problem_templates)
    issue = random.choice(negative_issues)
    customer_text = template[0].format(issue=issue)
    agent_text = template[1]
    
    conversation = [
        {"speaker": "customer", "text": customer_text},
        {"speaker": "agent", "text": agent_text}
    ]
    
    samples.append({
        "conversation_id": conversation_id,
        "conversation_json": json.dumps(conversation),
        "sarcasm": 0,
        "category": random.choice(categories),
        "sentiment": "negative"  # Real problems = negative
    })
    conversation_id += 1

# 3. Non-sarcastic + Positive (1500 samples)
for i in range(1500):
    template = random.choice(positive_templates)
    action = random.choice(positive_actions)
    customer_text = template[0].format(action=action)
    agent_text = template[1]
    
    conversation = [
        {"speaker": "customer", "text": customer_text},
        {"speaker": "agent", "text": agent_text}
    ]
    
    samples.append({
        "conversation_id": conversation_id,
        "conversation_json": json.dumps(conversation),
        "sarcasm": 0,
        "category": random.choice(categories),
        "sentiment": "positive"  # Gratitude = positive
    })
    conversation_id += 1

# 4. Non-sarcastic + Neutral (1000 samples)
for i in range(1000):
    template = random.choice(neutral_templates)
    customer_text = template[0]
    agent_text = template[1]
    
    conversation = [
        {"speaker": "customer", "text": customer_text},
        {"speaker": "agent", "text": agent_text}
    ]
    
    samples.append({
        "conversation_id": conversation_id,
        "conversation_json": json.dumps(conversation),
        "sarcasm": 0,
        "category": random.choice(categories),
        "sentiment": "neutral"  # Informational queries = neutral
    })
    conversation_id += 1

# Create dataframe
df = pd.DataFrame(samples)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_file = "data/ecommerce_conversations_clean.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Generated {len(df)} samples with CONSISTENT labels")
print(f"✅ Saved to: {output_file}")

# Show distribution
print(f"\n📊 Distribution:")
print(f"Sarcasm: {df['sarcasm'].value_counts().to_dict()}")
print(f"Sentiment: {df['sentiment'].value_counts().to_dict()}")
print(f"Category: {df['category'].value_counts().to_dict()}")

print("\n🎯 Key improvements:")
print("  ✓ Sarcastic text ALWAYS has negative sentiment (real-world accuracy)")
print("  ✓ Problem statements ALWAYS have negative sentiment")
print("  ✓ Gratitude ALWAYS has positive sentiment")
print("  ✓ Neutral queries have neutral sentiment")
print("  ✓ NO random sentiment assignment")
print("  ✓ Same text = same label (consistency)")
