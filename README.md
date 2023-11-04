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

------------------------------------------- TO BE UPDATED BY EVERYONE ------------------------------------------- 
 
## File Structure
This repository is organized into several directories and key files as follows:
- `/Resources/`: Raw CSV files of fake and true news articles used for analysis.
- `DataExploration.ipynb`: Jupyter notebook for initial data exploration and analysis.
- `SentimentAnalysis.ipynb`: Jupyter notebook for conducting sentiment analysis on the dataset.
- `DataModelling_AK.ipynb`: Notebook containing machine learning model for data prediction using RNN - LSTM model development.
- `DataModelling_JJ.ipynb`: Additional notebook with machine learning model for data prediction using RNN - LSTM model development.
- `/Output/`: Contains output files such as processed datasets or results from the analysis.
- `README.md`: Documentation explaining the project, its structure, and how to run it.

------------------------------------------- TO BE UPDATED BY EVERYONE ------------------------------------------- 

## Structure

### Backend

#### Exploratory Data Analysis (EDA)
The EDA process involved examining and summarizing the main characteristics of the dataset, often using visual methods. 
It provided a better understanding of the data's distribution and uncovered patterns, anomalies, and relationships between variables.
Exploratory Data Analysis was done in the following files: `DataExploration.ipynb` and  `SentimentAnalysis.ipynb`.

------------------------------------------- TO BE UPDATED BY JYOTSNA ------------------------------------------- 

##### Data Exploration (`DataExploration.ipynb`)

------------------------------------------- TO BE UPDATED BY JYOTSNA ------------------------------------------- 

##### Sentiment Analysis (`SentimentAnalysis.ipynb`) 
- **Objective**: To assess the emotional tone and subjectivity of news articles, differentiating between true and fake news.
- **Process**:
  - Loaded and cleaned articles from a cleaned CSV file containing both fake and true news, standardizing text for analysis.
  - Tokenized articles using spaCy and NLTK to break down text into individual words, removing stopwords and non-alphabetic characters.
  - Calculated word frequency distribution to identify the most common words in true and fake news.
  - Generated word clouds for a visual depiction of frequent words in each news type.
  - Analyzed polarity and subjectivity using TextBlob to evaluate the emotional content and the amount of personal opinion in articles.
  - Categorized articles based on polarity scores and compared sentiment distribution between true and fake news.
  - Plotted histograms to show the polarity distribution, revealing sentiment trends within articles.
  - Calculated and compared average polarity and subjectivity scores between true and fake news.
- **Findings**:
  - Both true and fake news articles generally have a slightly positive tone, with fake news showing a higher average polarity.
  - Fake news articles exhibit a higher level of subjectivity, indicating more opinion-based content.
  - A higher proportion of negatively toned articles were found in fake news compared to true news.
  - Politically charged words were often associated with negative sentiments in fake news.

------------------------------------------- TO BE UPDATED BY ASHA & JYOTSNA ------------------------------------------- 

#### Data Preprocessing for Modeling
Detailed steps on how the data was preprocessed for the machine learning models, including cleaning, feature extraction, and normalization.

#### Machine Learning Models
Explanation of the machine learning models developed to classify news articles, including the model types, training process, and evaluation metrics used.

------------------------------------------- TO BE UPDATED BY ASHA & JYOTSNA ------------------------------------------- 


------------------------------------------- TO BE UPDATED BY PRYJA & TARYN ------------------------------------------- 

### Frontend
- Interactive dashboard for user engagement and visualization
- Integration of Flask API for dynamic data handling
- Use of Tableau for data visualization components

## Setup and Installation


## Usage

------------------------------------------- TO BE UPDATED BY PRYJA & TARYN ------------------------------------------- 














------------------------------------------------- TO DELETE - OLD VERSION  ---------------------------------------------------------------------


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
 


### TO COMPLETE BY 5/11
Jyotsna - Update your tasks on the model - LSDM - RNN Model giving high accuracy, check against random inputs. Help Priya with Flask  <br>
Asha -  Update ReadME file, Correct EDA and Integrated Jyotsnas file, review model - need to complete further testing, 3 x trials <br>
Priya - Update ReadME file Today create Flask try to get input - random inputs in Python (backend) <br>
Taryn - Update ReadME file Interactive selections, control if statements Create wordcloud <br> 

### TO COMPLETE BY 6/11
Flask and front-end website ready, Tableau link, powerpoint presentation content ready to put into template 

------------------------------------------------- TO DELETE - OLD VERSION  ----------------------------------------------------------------------
