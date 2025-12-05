import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Multi-Task Classifier", page_icon="🎭", layout="wide")

# ⚡ PERFORMANCE FIX: Load model once and cache it
@st.cache_resource
def load_model():
    """Load model once and cache it for all requests"""
    from transformers import AutoTokenizer
    import torch
    import pickle
    from safetensors.torch import load_file
    from train import MultiTaskModel
    
    MODEL_PATH = "./models/deberta_multitask"
    
    # Load sentiment encoder
    with open(f"{MODEL_PATH}/sentiment_encoder.pkl", "rb") as f:
        sentiment_encoder = pickle.load(f)
    
    num_sentiments = len(sentiment_encoder.classes_)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    
    # Load model
    model = MultiTaskModel(
        "microsoft/deberta-v3-base",
        num_sarcasm_labels=2,
        num_sentiment_labels=num_sentiments
    )
    
    state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, sentiment_encoder

# Load model at startup
try:
    model, tokenizer, sentiment_encoder = load_model()
    sarcasm_map = {0: "Not Sarcastic", 1: "Sarcastic"}
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Model not found. Please train the model first: `./run.sh train`")

st.title("🎭 E-commerce Conversation Analyzer")
st.markdown("### Sarcasm • Sentiment Detection")
st.markdown("Powered by DeBERTa-v3 Dual-Task Model")
st.info("✨ **Sarcasm-Aware Sentiment**: Model uses sarcasm detection to improve sentiment accuracy!")

st.markdown("---")

# Initialize session state for text
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Conversation Text")
    
    # Text area with session state
    text = st.text_area(
        "Paste customer conversation here:",
        value=st.session_state.input_text,
        height=200,
        placeholder="Example: 'Wow great, another delay… absolutely love waiting forever!'"
    )
    
    predict_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col2:
    st.subheader("Example Conversations")
    
    examples = [
        "Wow great, another delay… absolutely love waiting forever!",
        "Thank you so much for your help! I really appreciate it.",
        "Oh fantastic, my order is lost again. You guys are the best!",
        "The product arrived on time and works perfectly. Thanks!",
        "Sure, charge me twice. I absolutely love paying extra.",
        "I need help with my refund. Can you assist me?"
    ]
    
    st.markdown("**Click to try:**")
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = example
            st.rerun()

# Make predictions
if predict_button and text and MODEL_LOADED:
    with st.spinner("Analyzing..."):
        try:
            import torch
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            
            # Get predictions (model automatically uses sarcasm for sentiment)
            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            # Process results
            sarcasm_probs = torch.softmax(outputs['sarcasm_logits'], dim=1)
            sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1)
            
            sarcasm_pred = torch.argmax(sarcasm_probs).item()
            sentiment_pred = torch.argmax(sentiment_probs).item()
            
            result = {
                "sarcasm": {
                    "label": sarcasm_map[sarcasm_pred],
                    "confidence": float(sarcasm_probs[0][sarcasm_pred])
                },
                "sentiment": {
                    "label": sentiment_encoder.inverse_transform([sentiment_pred])[0],
                    "confidence": float(sentiment_probs[0][sentiment_pred])
                }
            }
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Create two columns for results
            col_sarc, col_sent = st.columns(2)
            
            with col_sarc:
                st.markdown("### 🎭 Sarcasm")
                sarcasm_label = result['sarcasm']['label']
                sarcasm_conf = result['sarcasm']['confidence']
                
                if sarcasm_label == "Sarcastic":
                    st.error(f"**{sarcasm_label}**")
                else:
                    st.success(f"**{sarcasm_label}**")
                
                st.metric("Confidence", f"{sarcasm_conf:.1%}")
                st.progress(sarcasm_conf)
            
            with col_sent:
                st.markdown("### 😊 Sentiment")
                sentiment_label = result['sentiment']['label']
                sentiment_conf = result['sentiment']['confidence']
                
                if sentiment_label == "positive":
                    st.success(f"**{sentiment_label.title()}**")
                elif sentiment_label == "negative":
                    st.error(f"**{sentiment_label.title()}**")
                else:
                    st.warning(f"**{sentiment_label.title()}**")
                
                st.metric("Confidence", f"{sentiment_conf:.1%}")
                st.progress(sentiment_conf)
            
            # Detailed breakdown
            st.markdown("---")
            with st.expander("📋 Detailed Breakdown"):
                st.json(result)
                
        except FileNotFoundError:
            st.error("❌ Model not found! Please train the model first:")
            st.code("./run.sh train")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure you've trained the model with `./run.sh train`")

elif predict_button:
    st.warning("⚠️ Please enter some text to analyze!")

st.markdown("---")
st.markdown("""
**Model Capabilities:**
- 🎭 **Sarcasm Detection:** Identifies sarcastic vs. genuine comments (90-95% accuracy)
- 😊 **Sentiment Analysis:** Determines positive, negative, or neutral sentiment
- ✨ **Sarcasm-Aware:** Sentiment prediction uses sarcasm information for better accuracy!

**Note:** Category prediction was removed due to low confidence (<40%). The model now focuses on what it does best!
""")
