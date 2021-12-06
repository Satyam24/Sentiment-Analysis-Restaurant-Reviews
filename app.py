import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('clf_model.pkl', 'rb'))

# Creating function to perform preprocessing like converting to lowercase, word tokenization, removing special characters, stop words, punctuation and lemmatization.
stopWords = stopwords.words('english')

stopWords.remove('no')
stopWords.remove('not')

def transform_text(text):
    text_lower = text.lower()
    text_lower_token = nltk.word_tokenize(text_lower)
    text_final = []

    for i in text_lower_token:
        if i.isalnum():
            text_final.append(i)

    text_all = text_final[:]
    text_final.clear()

    for i in text_all:
        if i not in stopWords and i not in string.punctuation:
            text_final.append(i)

    text_new = text_final[:]
    text_final.clear()

    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [lemmatizer.lemmatize(word) for word in text_new]

    return " ".join(lemmatized_words)

st.title('Sentiment Analysis')

input_sent = st.text_area('Enter the restaurant review here...')

if st.button('Predict'):


    # Steps
    # 1. Preprocess
    transformed_sent = transform_text(input_sent)
    # 2. vectorize

    vector_input = vectorizer.transform([transformed_sent])
    # 3. predict

    result = model.predict(vector_input)[0]

    # 4. display result

    if result == 1:
        st.header("This review is Positive")
    else:
        st.header("This review is Negative")