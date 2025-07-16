import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from imblearn.over_sampling import RandomOverSampler
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Rest of your imports...
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


app = Flask(__name__)

# # Load your models first
# stand_scaler = pickle.load(open("bow.pkl", "rb"))
# loaded_model = pickle.load(open('nb.pkl', 'rb'))  # Changed variable name
# Example prediction
# Enhanced text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers (keeping basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Expand contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have",
        "'m": " am"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = BeautifulSoup(text, 'lxml').get_text()
    
    # Handle negations
    text = re.sub(r'\b(not|no|never)\s+(\w+)', r'\1_\2', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back to string
    text = ' '.join(tokens)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
def predict_sentiment(text):
    # Load model and vectorizer
    with open('best3_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer_3.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess and vectorize
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    confidence = max(probability[0])
    
    return f"Sentiment: {sentiment} (Confidence: {confidence:.2f})"
@app.route("/")
def home():
    return render_template("noob.html")


@app.route("/output", methods=["POST"])
def output():
    if request.method == "POST":
        try:
            # Get all form data
            MedInc = request.form["MedInc"]
            print(MedInc)

           
            res=predict_sentiment(MedInc)
            
            return render_template("noob.html", value=res)
            
        except Exception as e:
            return render_template("noob.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)