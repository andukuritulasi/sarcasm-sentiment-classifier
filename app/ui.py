import streamlit as st
from inference import predict_sarcasm

st.title("Sarcasm Detector for E-commerce Conversations")

text = st.text_area("Paste conversation here:", height=250)

if st.button("Predict"):
    result = predict_sarcasm(text)
    st.subheader("Prediction")
    st.write(f"**Sarcasm:** {result['label']}")
    st.write(f"**Confidence:** {result['confidence']:.2f}")
