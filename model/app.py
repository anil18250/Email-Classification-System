import streamlit as st
import pickle


st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧"
)


st.title("📧 Email Classification System using NLP")


model = pickle.load(
    open("model/email_model.pkl","rb")
)

vectorizer = pickle.load(
    open("model/vectorizer.pkl","rb")
)


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