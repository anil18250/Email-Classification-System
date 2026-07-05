import re

import nltk
from nltk.corpus import stopwords


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = [word for word in text.split() if word not in stopwords.words("english")]
    return " ".join(words)
