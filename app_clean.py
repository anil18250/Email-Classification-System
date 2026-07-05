import pickle
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
EMAIL_MODEL_PATH = MODEL_DIR / "email_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Email Classification System using NLP")
st.write("Classify emails into Spam, Work, Promotion, Social, or Normal.")

if not EMAIL_MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error("Model files are missing. Run `python train.py` first.")
    st.stop()

with EMAIL_MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with VECTORIZER_PATH.open("rb") as f:
    vectorizer = pickle.load(f)

subject = st.text_input("📌 Email Subject")
message = st.text_area("📝 Email Message", height=200)

if st.button("🔍 Check Email"):
    if subject.strip() == "" and message.strip() == "":
        st.warning("Please enter an email subject or message.")
    else:
        text = f"{subject} {message}".strip()
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max() * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction.lower() == "spam":
            st.error(f"🚨 Spam Email\n\nConfidence: {confidence:.2f}%")
        elif prediction.lower() == "work":
            st.info(f"💼 Work Email\n\nConfidence: {confidence:.2f}%")
        elif prediction.lower() == "promotion":
            st.warning(f"🏷️ Promotion Email\n\nConfidence: {confidence:.2f}%")
        elif prediction.lower() == "social":
            st.success(f"👥 Social Email\n\nConfidence: {confidence:.2f}%")
        elif prediction.lower() == "normal":
            st.success(f"✅ Normal Email\n\nConfidence: {confidence:.2f}%")
        else:
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence: {confidence:.2f}%")
