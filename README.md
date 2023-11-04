# Real vs Fake News Classification

Submitted for Monash University Bootcamp Project 4.

## Description
With the lines of truth being blurred between real and fake in the daily news, every day people becoming more reliant on social media for their news sources - it is more important than ever to be able to differentiate the two for reliable and accurate new sources. In light of this, we've built a straightforward and effective model using data analysis and machine learning to judge news stories. Our HTML website, powered by Flask, lets users easily check articles. Simply put in the text, hit submit, and get an immediate real-or-fake verdict.

## Dataset
Our analysis is grounded on a comprehensive dataset that separates authentic news from fake stories. We sourced our data from a widely-recognized collection available on Kaggle, which comprises two CSV files—one with verified news and the other with false reports. 

The key points about the dataset:
- **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle
- **Composition**: Two CSV files, one labeled 'Fake' and the other 'True'.
- **Data Fields**: Each record includes the title, text, subject, and date of the news article.
- **Volume**: The dataset provides a substantial number of articles, offering a robust foundation for training our machine learning model.

## Toolkit
- Python (pandas, pathlib, matplotlib, numpy, seaborn, Scikit-learn, nltk, spacy)
- Flask for web application backend
- Tableau for advanced data visualizations
- HTML, Javascript for frontend web development



------------------------------------------------- WORKING ON ---------------------------------------------------------------------

## Structure

### Backend
- Data cleansing and preprocessing for modeling
- Tokenization
- Word frequency analysis
- Sentiment analysis using NLP techniques
- Development of machine learning models to classify news articles

### Frontend
- Interactive dashboard for user engagement and visualization
- Integration of Flask API for dynamic data handling
- Use of Tableau for data visualization components

## Setup and Installation
Detail the required steps to install and run your project locally. This might include:


------------------------------------------------- WORKING ON ---------------------------------------------------------------------












------------------------------------------------- TO DELETE  ---------------------------------------------------------------------


### STRUCTURE

_Backend_<br>

Everyone - Create a basic data analysis and upload to github for EDA<br><br>

Jyotsna and Asha backend - cleanse data, modelling, creating csv for analysis -> for user recommendation, need user preference dataset<br>

- Cleaning with MongoDB(?)
- Limitization (remove special characters etc)
- Tokenization after cleaning
- Sentiment analysis - textblob and vader use natural language processing techniques
- Word frequency distribution
- Create a new file for each model
- Asha and Jyotsna can create a machine learning model and create test<br><br>

_Frontend_<br>
- Javascript, HTML, CSS<br>
- Flask API<br>
- Tableau<br>


Priya to create a dashboard, combine tableau with Javascript and connection through Flask
- Create multiple routes or single routes
- User interaction through Java, anything that needs interaction will be in the dashboard
- Combine Tableau wth Javascript and connection through Flask or using Teapot to input data<br>
- Taryn to create analysis in Tableau 
 

### TO COMPLETE 2/11
Asha - to create RNN or LSTM model
Jyotsna - to create RNN or LSTM model model
Priya - Create Flask API
- Gather user input for model
Save model in h5 -> In Python code for flask where it's reading the model and then (as it is used in the test file) preprocessing, limitization, use that code to predict
Taryn - Story/Tableau Visualization template

### TO COMPLETE BY 5/11
Jyotsna - Update your tasks on the model - LSDM - RNN Model giving high accuracy, check against random inputs. Help Priya with Flask  <br>
Asha -  Update ReadME file, Correct EDA and Integrated Jyotsnas file, review model - need to complete further testing, 3 x trials <br>
Priya - Update ReadME file Today create Flask try to get input - random inputs in Python (backend) <br>
Taryn - Update ReadME file Interactive selections, control if statements Create wordcloud <br> 

### TO COMPLETE BY 6/11
Flask and front-end website ready, Tableau link, powerpoint presentation content ready to put into template 

------------------------------------------------- TO DELETE  ---------------------------------------------------------------------
