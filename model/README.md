# 📧 Email Classification System using NLP

## Project Overview

Email Classification System is an NLP-based Machine Learning project that automatically classifies emails into:

- 🚨 Spam Email
- ✅ Normal Email (Ham)

The system uses Natural Language Processing techniques to analyze email text and predict the category.

---

## Features

✅ Email text preprocessing  
✅ Tokenization and cleaning  
✅ TF-IDF feature extraction  
✅ Machine Learning classification  
✅ Streamlit Web UI  
✅ Spam detection with confidence score  

---

## Project Structure

```
Email Classification System

│
├── app.py
│   └── Streamlit User Interface
│
├── train.py
│   └── Model training file
│
├── preprocess.py
│   └── NLP text cleaning
│
├── dataset
│   └── emails.csv
│
├── model
│   ├── email_model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
│
└── README.md
```

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK

---

## Installation

Clone the project:

```bash
git clone your_repository_link
```

Go to project folder:

```bash
cd Email-Classification-System
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Train Model

Run:

```bash
python train.py
```

Output:

```
Training Completed
```

It creates:

```
model/
 ├── email_model.pkl
 └── vectorizer.pkl
```

---

## Run Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## User Interface

The application contains:

1. Email Subject input
2. Email Message box
3. Check Email button
4. Prediction result

Example:

Input:

```
Congratulations you won free money
```

Output:

```
🚨 Spam Email Detected
```

---

## Machine Learning Workflow

```
Dataset
   |
   ↓
Text Cleaning
   |
   ↓
NLP Preprocessing
   |
   ↓
TF-IDF Vectorization
   |
   ↓
Naive Bayes Model
   |
   ↓
Prediction
```

---

## Future Improvements

- Deep Learning model integration
- BERT based classification
- Phishing email detection
- Email priority prediction
- Cloud deployment

---

## Author

Email Classification System using NLP
