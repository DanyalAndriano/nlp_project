###########################################

Author: Danyal Andriano

Email:  [danyal.andriano@gmail.com](mailto:danyal.andriano@gmail.com)

###########################################

# Online Reviews - Automated Insights & Understanding 

The objective of this project was to research, design and help deploy natural language processing solutions to:
1) Automate responses to restaurant reviews online.
2) Extract information to gain insights into products and topics being spoken about. 
------------------------------------------------
Below is a list of notebooks in the directory and the components of the project included in each. Use the links to render in html (recommended to view hyperlinks and graphics). 

_NOTE: Data for this notebook is not open source and is not included in this directory._

# Part 1: Automation - NLP/ NLU

## Data Collection, Preparation & Preprocessing
> [Data Preparation & Preprocessing](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/data_preparation_and_preprocessing.html) (MySQL, Data Wrangling, Feature Engineering, Text Preprocessing)
<br>

The data consisted of 140 related databases in MySQL Workbench (restored from an S3 bucket in AWS). All potentially relevant columns were queried and read into pandas dataframes. 

Four tables were created: 
<br>
**Reviews**: review text and associated information (such as store, platform, user rating).
<br>
**Categories**: human rated categories and corresponding positive/ negative sentiment.
<br>
**Locations**: location information for each restaurant.
<br>
**Category Info**: additional information about categories (e.g., label, subcategory). 

Labels
----------------
Sentiment: `negative`, `positive` and `mixed` labels were created from existing negative and positive topic labels. If a review had both negative and positive topics, it was labeled _mixed_. 

<img src="https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/category_labels.png" width="800"> 

Reviews were predominantly `positive`, with `mixed` and `negative` minority classes.

![Sentiment Labels](https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/sentiment_label_distribution.png) 

Text Preprocessing 
------------------
Text was preprocessed and saved for later use.

<img src="https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/pandas_preprocessed.png"> 

----------------------------------------

## Feature Engineering & Sentiment Analysis

> [Unsupervised Sentiment & Text Features Exploration](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/features_benchmarking_eda.html) (Feature Engineering, Resampling Methods/ Class-Imbalance)
>
> [N-gram XGBoost Pipeline](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/sentiment-analysis-ML.html) (Tf-Idf, Bag-of-words, Feature Selection, Optimal Error Rate, Sklearn Pipelines, XGBoost, Fine-Tuning)

Why bother with sentiment analysis when user ratings accompany review text? In general, user ratings were only moderately predictive of the sentiment labels and are therefore unreliable for inference. Furthermore, a 5 star review may still contain mixed sentiment - an automated system would need to be sensitive to this mixed sentiment so that the response, as well as insights into how customers _feel_ about a brand and their _opinions_ of the brand, is appropriate.  

![ratings](https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/user_ratings_freq.png) ![ratings predictions](https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/ratings_preds.png)

The `ratings`, `unsupervised sentiment scores` - from Amazon Comprehend, Textblob and VaderSentiment - as well as `text features` were combined as inputs to various models. Oversampling, undersampling and class-weights were all used to try and better balance the predictions.

<img src=https://github.com/DanyalAndriano/nlp_project/blob/master/resampling_pipeline.png>

## End-to-end Sentiment Classification with Bert
>[Fast-BERT Sentiment Analysis]() (Fast-Bert Library, Transfer Learning in NLP, Error Analysis, Noisy Labels, Calibration)

## Deployment & Production
>[Cost-Benefit Analysis]() (Business Value and Risk, Decision Management)
><br>
>[Fast-Bert Sentiment Analysis]() (Production, API model deployment, Cloud Services)
><br>
>[XGBoost Inference Pipeline]() (Custom Preprocessing with Sklean in AWS, Inference Pipelines)

# Part 2: Insight - Location Based Sentiment Analysis
## Data Mining & Visualization
>[Menu Analytics]() (Case Study, SpaCy NLP Pipelines, Entity Recognition, Data Viz (Time Series, Geospatial)
><br>
>[Unsupervised Topic Modeling]() (Text Mining, Text Summarization)

## Tableau Dashboard
> [Menu Items & Sentiment in Reviews]() (Data Visualization, Data Mining, Interactive Dashboards, BI)

<img src="https://github.com/DanyalAndriano/nlp_reviews_solutions/blob/master/graphics/Screenshot%20(27).png" width="1000">

# Part 3: Next Steps
## Entity Recognition Models
 
## SpaCy Pipelines 
- build comprehensive pipelines in AWS to combine classification and entity-recognition models into a single process. 

## Response Automation

# Project Overview
- Labels engineered.
<br>
- Ratings are not accurate enough to determine sentiment - model.
<br>
- Two approaches - classic machine learning with engineered features; transfer learning.
<br>
- end-to-end bert model fine-tuned for classification performed best; simplest solution. 
<br>
- reached optimal error rate based on human performance.
<br>
- deployed into production and used in conjunction with text extraction to gain insight.  











NOTEBOOK 7 - MENU ANALYTICS WITH SpaCy
NOTEBOOK 8 & 9 - AWS DEPLOYMENT
TABLEAU DASHBOARD 

