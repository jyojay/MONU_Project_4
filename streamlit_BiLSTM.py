# Import dependencies

import numpy as np
# import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
# from sklearn.feature_extraction.text import TfidfVectorizer  
import spacy
from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Setting page configuration
st.set_page_config(page_title="Fake News Detector")

#loading  spacy
@st.cache_resource 
def load_nlp(path):
    nlp = spacy.load(path)
    return nlp

# loading model
@st.cache_resource  # This decorator ensures the function runs only once
def load_my_model(path):
    loaded_model = load_model(path) 
    return loaded_model

#loading data_source
@st.cache_resource 
def load_df(path):
    news_df = pd.read_csv(path)
    news_df = news_df.drop(['Unnamed: 0'], axis=1)
    news_df.dropna(inplace = True)
    return news_df

# Fitting Vectorizer 
@st.cache_resource
def load_token(news_df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['text'])
    return tokenizer

nlp = load_nlp(r"C:\Users\priya\en_core_web_sm-3.5.0")
loaded_model= load_my_model('models/model_biLSTM.h5')
news_df=load_df('Resources/nlp_cleaned_news.csv')
tokenizer = load_token(news_df)

def prediction(input_text):
    
    # Cleaning user input data
    clean_text = data_cleaning(input_text)
    clean_text_ser = pd.Series(clean_text)

    # Vectorizing cleaned data
    vectorized_input_data = tokenizer.texts_to_sequences(clean_text_ser)

    # Padding data
    pad_input_data = pad_sequences(vectorized_input_data, maxlen = 4456, padding = 'post')

    # Predicting outcome
    prediction = loaded_model.predict(pad_input_data)
    
    return prediction[0]

# User defined function definition

def data_cleaning(text):

    # changing to lower case
    text = text.lower()

    # Getting document ready for NLP tasks
    doc = nlp(text)

    # Empty list for storing cleaned data
    clean_text = ""

    # Remove stop words and lemmatize
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Join the lemmas back into a string
    clean_text = ' '.join(lemmas)

    return(clean_text)

# ----------------------------------------------------------------------
# WEB Site Design Using Streamlit 
# ----------------------------------------------------------------------

#User define function definition
def clear():
    st.session_state['name'] = ' ' 
  
# WebPage Title
st.title("Fake News Detector")

# Link to direct Tableau Visualisations

st.write("To check on our Analysis and Visualisations :  [Click Here](https://public.tableau.com/views/P4_16990786163050/Homepagedash?:language=en-GB&publish=yes&:display_count=n&:origin=viz_share_link)")

st.subheader("Enter Your News")

input_text = st.text_input("Enter your news text below and press 'Enter' to receive the classification result.",key='name')

st.button('Clear', on_click=clear)

if input_text:
    if input_text== " ":
        pred=2
    else:
        pred = prediction(input_text)   
    
    # Printing outcome based on predicted value
    if pred < 0.5:
        st.subheader("News :")
        st.write(input_text)
        st.subheader("This news is classified as FAKE ",divider ='red')
    elif pred >= 0.5 and pred <=1:
        st.subheader("News :")
        st.write(input_text)
        st.subheader("This news is classified as TRUE.",divider = 'red')
    
    else:
        st.write(" ")
