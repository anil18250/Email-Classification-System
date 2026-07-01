import pandas as pd
import pickle
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "emails.csv"
MODEL_DIR = BASE_DIR


# Load data

data = pd.read_csv(DATA_PATH)



# Cleaning

data["email"] = data["email"].apply(
    clean_text
)



X = data["email"]

y = data["category"]



# Vectorization

vectorizer = TfidfVectorizer()


X = vectorizer.fit_transform(X)



# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train

model = MultinomialNB()


model.fit(
    X_train,
    y_train
)



# Save

with open(MODEL_DIR / "email_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(MODEL_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)



print("Training Completed")