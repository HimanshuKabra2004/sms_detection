import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📩 SMS Spam Detector")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Preprocess
    cleaned = preprocess(input_sms)
    # Vectorize
    vector_input = vectorizer.transform([cleaned])
    # Predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("❌ Spam SMS")
    else:
        st.success("✅ Not a Spam (Ham)")
