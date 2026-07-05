import pandas as pd
import pickle
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


import sys
import pandas as pd
import pickle
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Make sure the project root is on sys.path so we can import preprocess from
# the repository root when running model/train.py directly.
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess import clean_text

DATA_PATH = ROOT_DIR / "dataset" / "emails.csv"
MODEL_DIR = BASE_DIR


# Load data
data = pd.read_csv(DATA_PATH)


# Normalize column names to lowercase for detection
cols = [c.lower().strip() for c in data.columns]

# If CSV has separate subject and message columns, combine them;
# otherwise try common single-column names.
if "subject" in cols and "message" in cols:
    subj = data.columns[cols.index("subject")]
    msg = data.columns[cols.index("message")]
    data["email"] = (data[subj].fillna("") + " " + data[msg].fillna("")).astype(str)
else:
    # candidates for a single text column
    for candidate in ("email", "text", "message", "body", "content", "subject"):
        if candidate in cols:
            data["email"] = data[data.columns[cols.index(candidate)]].astype(str)
            break
    else:
        raise KeyError(f"No suitable text column found in {list(data.columns)}")


# Determine label/target column
for candidate in ("category", "label", "target", "class"):
    if candidate in cols:
        label_col = data.columns[cols.index(candidate)]
        break
else:
    # fallback: try common names with capitalization variations
    raise KeyError(f"No label column found in {list(data.columns)}")


# Cleaning
data["email"] = data["email"].apply(clean_text)


X = data["email"]
y = data[label_col]



# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)



# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train
model = MultinomialNB()
model.fit(X_train, y_train)



# Save
with open(MODEL_DIR / "email_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(MODEL_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


print("Training Completed")