import streamlit as st
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "email_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧"
)


st.title("📧 Email Classification System using NLP")


if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error("Model files are missing. Please train the model first.")
    st.stop()


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


subject = st.text_input("Email Subject")

message = st.text_area(
    "Email Message",
    height=200
)


if st.button("Check Email"):

    text = subject + " " + message

    vector = vectorizer.transform([text])

    result = model.predict(vector)


    if result[0] == "spam":
        st.error("🚨 Spam Email Detected")

    else:
        st.success("✅ Normal Email")