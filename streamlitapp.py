import streamlit as st
import joblib
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# # Load the saved models and vectorizers
# with open("vectorizer_bow.pkl", "rb") as f1:
#     vectorizer_bow = pickle.load(f1)
    
# with open("vectorizer_tfidf.pkl", "rb") as f2:
#     vectorizer_tfidf = pickle.load(f2)
    
    
vectorizer_bow = joblib.load('vectorizer_bow.pkl')
vectorizer_tfidf = joblib.load('vectorizer_tfidf.pkl')
with open('vectorizer_word2vec.pkl', 'rb') as f3:
    vectorizer_word2vec = pickle.load(f3)

logistic_model = joblib.load('logistic_model.pkl')
# Load model
# with open("logistic_model.pkl", "rb") as f4:
#     logistic_model = pickle.load(f4)

# Function to preprocess input text
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove characters other than alphabets and numerics
    text = re.sub("[^A-Za-z0-9 ]+", " ", text)
    
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    
    # Remove hyperlinks
    text = re.sub(r'(http|https|ftp|ssh)://[\w_-]+(?:\.[\w_-]+)+[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-]?', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    text= " ".join([WordNetLemmatizer().lemmatize(y) for y in text.split()])
    
    return text

def avgword2vec(tokens,model):
    avg_vec=[]
    for review in tokens:
        vectors=[model.wv[word] for word in review if word in model.wv]
        if vectors:
            avg_vec.append(sum(vectors)/len(vectors))
        else:
            avg_vec.append([0] * model.vector_size)
    return avg_vec
# Function to vectorize input text
def vectorize_text(text, method):
    if method == 'Bag of Words':
        return vectorizer_bow.transform([text]).toarray()
    elif method == 'TF-IDF':
        return vectorizer_tfidf.transform([text]).toarray()
    elif method == 'Word2Vec':
        # Assuming Word2Vec returns a single vector
        tokens=text.split()
        
        return avgword2vec(tokens,vectorizer_word2vec)

# Streamlit App
st.title("Amazon Kindle Review Sentiment Analysis")
st.write("Choose a text vectorization method and enter your review to predict sentiment.")

# User Input
text = st.text_area("Enter your review text here:")
method = st.selectbox("Choose a vectorization method:", ['Bag of Words', 'TF-IDF'])

if st.button("Predict Sentiment"):
    if not text.strip():
        st.error("Please enter a review!")
    else:
        try:
            # Preprocess the text
            processed_text = preprocess(text)
            
            # Vectorize the text based on user choice
            text_vector = vectorize_text(processed_text, method)
            
            # Predict sentiment using the logistic model
            prediction = logistic_model.predict(text_vector)
            
            # Output the result
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.success(f"The predicted sentiment is: {sentiment}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
