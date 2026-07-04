import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("dataset/emails.csv")

# Combine subject and message columns
df["text"] = (df["subject"].fillna("") + " " + df["message"].fillna("")).str.strip()

X = df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save model
with open("model/email_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ vectorizer.pkl created successfully")