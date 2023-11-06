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

------------------------------------------- TO BE UPDATED BY EVERYONE ------------------------------------------- 

## Toolkit
- Python (pandas, pathlib, matplotlib, numpy, seaborn, Scikit-learn, nltk, spacy)
- Flask for web application backend
- Tableau for advanced data visualizations and data analysis
- HTML, Javascript for frontend web development
 
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
It provided a better understanding of the data's distribution and uncovered patterns, anomalies, and relationships between variables. </br>
The EDA was conducted in two distinct Jupyter notebooks: `DataExploration.ipynb` and `SentimentAnalysis.ipynb`.  </br>
`DataExploration.ipynb` focused on initial data exploration, including data cleaning and standardization, while `SentimentAnalysis.ipynb` delved deeper into the emotional tone of the articles through sentiment analysis, utilizing natural language processing techniques to assess the polarity and subjectivity of the text.

------------------------------------------- TO BE UPDATED BY TARYN ------------------------------------------- 

##### Data Exploration (`DataExploration.ipynb`)

------------------------------------------- TO BE UPDATED BY TARYN ------------------------------------------- 

##### Sentiment Analysis (`SentimentAnalysis.ipynb`) 
- **Objective**: To assess the emotional tone and subjectivity of news articles, differentiating between true and fake news.
- **Process**:
  - Loaded and cleaned articles from a cleaned CSV file containing both fake and true news, standardizing text for analysis.
  - Tokenized articles using spaCy and NLTK to break down text into individual words, removing stopwords and non-alphabetic characters.
  - Calculated word frequency distribution to identify the most common words in true and fake news.
  - Generated word clouds for a visual depiction of frequent words in each news type.
    
    ![1](https://github.com/jyojay/MONU_Project_4/assets/134185577/4fca347b-ca77-4758-bf92-2146835aa456)
    
  - Analyzed polarity and subjectivity using TextBlob to evaluate the emotional content and the amount of personal opinion in articles.
  - Categorized articles based on polarity scores and compared sentiment distribution between true and fake news.

    ![2](https://github.com/jyojay/MONU_Project_4/assets/134185577/bb10da28-03c8-4651-80be-4843742af99f)

  - Plotted histograms to show the polarity distribution, revealing sentiment trends within articles.
 
    ![3](https://github.com/jyojay/MONU_Project_4/assets/134185577/240b32ac-5e2e-4b92-9b99-f12bb7f47652)

  - Calculated and compared average polarity and subjectivity scores between true and fake news.
 
    ![4](https://github.com/jyojay/MONU_Project_4/assets/134185577/7087c3f1-31d9-43fa-ac65-f69edd0b5fe1)
    
- **Findings**:
  - Both true and fake news articles generally have a slightly positive tone, with fake news showing a higher average polarity  
  - Fake news articles exhibit a higher level of subjectivity, indicating more opinion-based content.
  - A higher proportion of negatively toned articles were found in fake news compared to true news.
  - Politically charged words were often associated with negative sentiments in fake news ("trump", "clinton", "obama", "police", "media").
  
   ![5](https://github.com/jyojay/MONU_Project_4/assets/134185577/b4324c5f-c4d8-4f7d-ae1e-b64827fe6d35)

### Data Modelling   

#### Machine Learning Models
##### Data Preperation for model training
- 

#### Newural Network Models

#### Limitations

------------------------------------------- TO BE UPDATED BY PRYJA & TARYN ------------------------------------------- 

### Frontend
- Interactive dashboard for user engagement and visualization
- Integration of Flask API for dynamic data handling
- Use of Tableau for data visualization components and comprehensive analysis

## Setup and Installation
- 
-
-
-
- For data visualisation involved further cleansed data into smaller filetypes for use in Tableau, such that irrelevant data was removed and columns were relabelled for ease of comprehension
- Designed each worksheet with a visualisation taken from the clean data sets and compiled into relevant dashboards
- Created dashboards for each analysis and model created and differentiated data from 
- Wrote a comprehensive and detailed data analysis per each visualisation and an intepretation of the results
- Compiled all dashboards into a story to be linked to the website incorporating the model
- Designed color theme, imagery and stylization.


## Usage

------------------------------------------- TO BE UPDATED BY PRYJA & TARYN ------------------------------------------- 
Screenshot of html page (**THIS IS JUST A PLACEHOLDER**)
<img width="1300" alt="Website image" src="https://github.com/jyojay/MONU_Project_4/blob/e2137b115879e6c0b7127c694249c770a8ce8081/ToRemove%20(Working%20files)/Taryn%20Fordyce/real_news_sentiment.png"><br><br>

The word analysis page created in Tableau using a variety of visual techniques including word cloud analysis, sentiment analysis and a comprehensive breakdown of the visulisations<br>

The link can be be found [here](https://public.tableau.com/views/P4_16990786163050/Homepage?:language=en-GB&publish=yes&:display_count=n&:origin=viz_share_link)



<img width="1440" alt="Tableau data analysis image" src="https://github.com/jyojay/MONU_Project_4/blob/ac061a849b0c0064e2dec6e8a02635e56a8bef3a/ToRemove%20(Working%20files)/Taryn%20Fordyce/tableau%20page.png">


