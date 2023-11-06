# Import dependencies

import numpy as np
import pickle
import spacy
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer  


# Loading Machine Learning model and other recources

nlp = spacy.load(r"C:\Users\priya\en_core_web_sm-3.5.0")
loaded_model =pickle.load(open('models/model_svm2.pkl','rb'))

# Loading data-CSV file 

news_df = pd.read_csv('Resources/nlp_cleaned_news.csv')
news_df = news_df.drop(['Unnamed: 0'], axis=1)
news_df.dropna(inplace = True)

# Fitting Vectorizer 

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(news_df['text'])

# User define function definitions 

def prediction(input_text):
    
    # Cleaning user input data
    clean_text = data_cleaning(input_text)
    clean_text_ser = pd.Series(clean_text)

    # Vectorizing cleaned data
    vectorized_input_data = Tfidf_vect.transform(clean_text_ser)

    # Predicting outcome
    prediction = loaded_model.predict(vectorized_input_data)
    
    return prediction[0]

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


# Setting page configuration

st.set_page_config(page_title="Fake News Detector")

#User define function definition
def clear():
    st.session_state['name'] = ' ' 
    
# Website Title

st.title("Fake News Detector")

# Link to direct Tableau Visualisations

st.write("To check on our Analysis and Visualisations :  [Click Here](https://public.tableau.com/views/P4_16990786163050/Homepage?:language=en-GB&publish=yes&:display_count=n&:origin=viz_share_link)")

st.subheader("Enter Your News")

input_text = st.text_input("Copy + paste your news and hit 'Enter' to get your result",key='name')

st.button('Clear', on_click=clear)

if input_text:
    if input_text== " ":
        pred=2
    else:
        pred = prediction(input_text)   
    
    # Printing outcome based on predicted value
    if pred == 0:
        st.subheader("News :")
        st.write(input_text)
        st.subheader("OOPS.... Fake News !!!",divider ='red')
    elif pred==1:
        st.subheader("News :")
        st.write(input_text)
        st.subheader("You have got Real News !!!",divider = 'red')
    else:
        st.write(" ")
    
    
