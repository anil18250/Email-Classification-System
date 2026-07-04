import streamlit as st
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Classification System using NLP")
st.write("Classify emails into Spam, Normal, Work, Promotion and Social.")

# -------------------------------
# Load Model
# -------------------------------
with open("model/email_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# User Input
# -------------------------------
subject = st.text_input("Email Subject")

message = st.text_area(
    "Email Message",
    height=200
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Check Email"):

    if subject.strip() == "" and message.strip() == "":
        st.warning("Please enter an email subject or message.")
    else:

        text = subject + " " + message

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        confidence = model.predict_proba(vector).max() * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction.lower() == "spam":
            st.error(f"🚨 Spam Email\n\nConfidence : {confidence:.2f}%")

        elif prediction.lower() == "normal":
            st.success(f"✅ Normal Email\n\nConfidence : {confidence:.2f}%")

        elif prediction.lower() == "work":
            st.info(f"💼 Work Email\n\nConfidence : {confidence:.2f}%")

        elif prediction.lower() == "promotion":
            st.warning(f"🏷️ Promotion Email\n\nConfidence : {confidence:.2f}%")

        elif prediction.lower() == "social":
            st.success(f"👥 Social Email\n\nConfidence : {confidence:.2f}%")

        else:
            st.write(f"Prediction : {prediction}")
            st.write(f"Confidence : {confidence:.2f}%")