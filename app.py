import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

model = pickle.load(open('word2vec_model.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))   

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def tweet_to_vector(tweet):
    words = tweet.split()
    vectors = []
    for word in words:
        try:
            vectors.append(model[word])
        except KeyError:
            # Ignore out-of-vocabulary words
            pass
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)



def predict_hate_speech(model_path, tweet):
    
    with open(model_path, 'rb') as f:
        model2 = pickle.load(f)
    
    tweet=stemming(tweet)
    vector = tweet_to_vector(tweet)
    prediction = model2.predict(vector.reshape(1, -1))
    return prediction



if __name__ == '__main__':
    jadoo = 'model.pkl'
    st.title('Twitter hate speech detection app ')
    st.subheader("Input the Tweets below")
    sentence = st.text_area("Enter your tweet  here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=predict_hate_speech(jadoo,sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('not hate')
        if prediction_class == [1]:
            st.warning('offensive')
        if prediction_class == [2]:
            st.warning('hate')