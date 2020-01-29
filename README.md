###########################################

Author: Danyal Andriano

Email:  [danyal.andriano@gmail.com](mailto:danyal.andriano@gmail.com)

###########################################

# Online Reviews - Automated Insights & Understanding 

The objective of this project was to research, design and help deploy natural language processing solutions to:
1) Automate responses to restaurant reviews online.
2) Extract information to gain insights into products and topics being spoken about. 

Below is a list of notebooks in the directory and the components of the project included in each. Use the links to render in html (recommended to view hyperlinks and graphics). 

_NOTE: Data for this notebook is not open source and is not included in this directory._

# Part 1: Automation - NLP/ NLU
------------------------------------------------

# Data Collection, Preparation & Preprocessing
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
Reviews were predominantly `positive`, with `mixed` and `negative` minority classes.

![Sentiment Labels](https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/sentiment_label_distribution.png) 

Text Preprocessing 
------------------
Text was preprocessed and saved for later use.

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/pandas_preprocessed.png"> 

----------------------------------------

# Feature Engineering & Sentiment Analysis

> [Unsupervised Sentiment & Text Features Exploration](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/features_benchmarking_eda.html) (Feature Engineering, Resampling Methods/ Class-Imbalance)
>
> [N-gram XGBoost Pipeline](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/sentiment-analysis-ML.html) (Tf-Idf, Bag-of-words, Feature Selection, Optimal Error Rate, Sklearn Pipelines, XGBoost, Fine-Tuning)

Why bother with sentiment analysis when user ratings accompany review text? In general, user ratings were only moderately predictive of the sentiment labels and are therefore unreliable for inference. Furthermore, a 5 star review may still contain mixed sentiment - an automated system would need to be sensitive to this mixed sentiment so that the response, as well as insights into how customers _feel_ about a brand and their _opinions_ of the brand, is appropriate.  

![ratings](https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/user_ratings_freq.png) ![ratings predictions](https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/ratings_preds.png)

Sentiment & Text Features
-------------------------------
The `ratings`, `unsupervised sentiment scores` - from Amazon Comprehend, Textblob and VaderSentiment - as well as `text features` were combined as inputs to various models. A pipeline was used to test numerous models and resampling techniques (oversampling, undersampling and class-weights) to find the best performing model. Prediction of the minority class is particularly important, as this is a cost-sensitive classification problem (i.e., misclassifying mixed sentiment is costly and carries business risks).

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/resampling_pipeline.png" width="750">

The `Amazon Comprehend sentiment` scores contributed the most predictive value, followed by `Textblob`, and then `VaderSentiment` and `Text Features`.

<img src=https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/sentiment_feature_importance.png width="750">

Ngrams
-----------------
Features for language modeling/ text classification are often created by mapping words or phrases to vectors of real numbers. The values of the vectors depend on the method used to weight the word/phrase. Three different methods are used for predicting sentiment: '`TF-IDF` (tvec), `count vectorization` (cvec) and `word embeddings`.   

`Logistic Regression` and `XGBoost` classifiers were be used to benchmark the tvec and cvec ngrams. 

Logistic Regression and XGBoost classifiers will be used to benchmark the tvec and cvec ngrams.

<img src=https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/ngram_cvec.png width="650">

XGBoost Pipeline
-----------------------
The `TF-IDF Trigrams` resulted in the best `macro-weighted F1-score` for the XGBoost model. A pipeline containing all sentiment, text and tf-idf features was used with `class-balanced` XGBoost. Performance was improved compared with sentiment and text features alone.

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/multiclass_ROC_xgboost.png" width="500"> <img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/xgboost_cm.png" width="350">

# End-to-end Sentiment Classification with Bert
>[Fast-BERT Sentiment Analysis](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/notebooks/bert_sentiment_classification.html) (Fast-Bert Library, Transfer Learning in NLP, Error Analysis, Noisy Labels, Calibration)

The fast-bert library was used to fine-tune pytorch transfomer's bert language model. Fast-bert is a Fast-AI inspired high level wrapper for the transformer architectures that works particularly well for fine-tuning these models to downstream classification tasks. The pytorch transformer models are smaller and faster than the original bert architecture, making them especially good for production. 

<img src=https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/pytorch_transformers.png width="750"> 

Performance
-------------------------
The fast-bert model performed better than XGBoost, and required less preprocessing. The fast-bert framework was added to the AWS ecosystem and the classifier was made available as an endpoint. 

<img src=https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/bert_cm_uncorrected.png width="400">

Can we Trust the Uncertainty in the Predictions?
-------------------------------------------------
Probabilistic classifiers output a probability distribution over target classes, rather than just a prediction. The value of this distribution is that it represents the amount of uncertainty in those predictions. However, the probabilities outputted do not always accurately represent the amount of uncertainty - sometimes probabilities are over- or under-confident.

**["Calibration - the idea that a model's predicted probabilities of outcomes reflect true probabilities of those outcomes - formalizes this notion"](https://arxiv.org/abs/1904.01685)** 

>**NOTE:** Custom methods were created for this multiclass calibration. See: `calibration_methods.py` and `calibration_functions.py`.
Methods from the [Netcal](https://fabiankueppers.github.io/calibration-framework/build/html/index.html) and [Calibration](https://github.com/markus93/NN_calibration) libraries were used directly or adapted. Certain methods from this repo were also adapted [NN Calibration](https://github.com/markus93/NN_calibration). 

While overall the model was well calibrated, classwise-reliability diagrams showed that the `positive` and `mixed` classes are miscalibrated. Isotronic Regression helped to correct this miscalibration. 

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/uncalibrated_classwise.png" width="400"> <img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/iso_calibration.png" width="400">

Calibration methods were further evaluated with calibration error metrics (ECE, classwise-ECE, Adaptive ECE), and proper loss metrics (Brier and Negative Log-Likelihood). `Isotronic Regression` and `Dirichlet scaling` appeared to best correct miscalibration. 

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/calibration_evaluation_metrics.png">
  
  
# Deployment & Production
>[Cost-Benefit Analysis](https://nbviewer.ipython.org/github/DanyalAndriano/nlp_project/blob/master/notebooks/cost_benefit_analysis.html) (Business Value and Risk, Decision Management)
>
>[Fast-Bert](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/aws/fast-bert-sentiment_aws.html) (AWS SageMaker, Custom DL Frameworks, Production, API model deployment)
><br>
>[XGBoost Inference Pipeline](http://htmlpreview.github.io/?https://github.com/DanyalAndriano/nlp_project/blob/master/aws/inference-pipeline-xgboost-sentiment.html) (Custom Preprocessing with Sklean in AWS, Inference Pipelines)

An initially high confidence cutoff was used (.95) to try and reduce misclassifications. This cutoff will still allow responses to be automated for 88% of the reviews. 

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/cost_benefit.png">

# Part 2: Insight - Location Based Sentiment Analysis
------------------------------------------------
## Data Mining & Visualization
>[Menu Analytics](https://nbviewer.ipython.org/github/DanyalAndriano/nlp_project/blob/master/notebooks/menu_analytics.html) (Case Study, SpaCy NLP Pipelines, Entity Recognition, Data Viz (Time Series, Geospatial)
><br>
>[Unsupervised Topic Modeling](Will be added soon) (Text Mining, Text Summarization)

Menu Extraction
---------------------
`SpaCy` was used to extract menu items from unstructured text. Information about the menu items and sentiment of reviews was used to create a dashboard, and a number of other visualizations (see notebook) to gain insights.

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/menu_extraction.png">

## Tableau Dashboard
> [Menu Items & Sentiment in Reviews](https://public.tableau.com/profile/danyal.andriano#!/vizhome/Menu_Analytics_Dashboard/MenuAnalyticsDashboard?publish=yes) (Data Visualization, Data Mining, Interactive Dashboards, BI)

<img src="https://github.com/DanyalAndriano/nlp_project/blob/master/graphics/Screenshot%20(27).png" width="1000">

# Part 3: In progress...

> 1) Redflags Classifier (detect insults, disrimination, injury, illness, theft) 
> 2) Entity Recognition Model + Categorization (menu items -> food subcategories)
> 3) SpaCy Pipelines 
>     - build comprehensive pipelines in AWS to combine classification, entity-recognition, rule-based filtering, and fuzzy matching into a single process. 
> 4) Full Response Automation

