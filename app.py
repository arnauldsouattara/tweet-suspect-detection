import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

st.title(" Détection de Tweets Suspects")
st.markdown("Entrez un tweet pour déterminer s'il est **suspect** ou non.")

tfidf, model = load_models()

tweet = st.text_area(" Votre tweet :", height=150)

if st.button("Analyser"):
    if tweet.strip() == '':
        st.warning("Veuillez entrer un tweet.")
    else:
        cleaned = preprocess(tweet)
        X = tfidf.transform([cleaned])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        if prediction == 0:
            st.error(f" Tweet SUSPECT (confiance : {proba[0]*100:.1f}%)")
        else:
            st.success(f" Tweet NON SUSPECT (confiance : {proba[1]*100:.1f}%)")