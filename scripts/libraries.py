# python libraries
import math
import os
from pathlib import Path
import random
import re
import sys
from time import time
import zipfile

# general libraries
import collections
from collections import Counter
from functools import reduce
import itertools
import json
import mysql.connector
import numpy as np
import pandas as pd
import pandas_bokeh
import requests
from requests.auth import HTTPBasicAuth

# geospatial libraries
#import geopandas as gpd
#import geopandas.tools
#import plotly.express as px
#import plotly.graph_objects as go
#from pygeocoder import Geocoder
#from shapely.geometry import Point, Polygon

# NLP libraries
import gensim
from gensim import corpora
from gensim.models import Word2Vec, Phrases
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.wrappers import LdaMallet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import wordninja

import nltk
from nltk.corpus import stopwords, wordnet, brown, reuters 
from nltk import collocations
from nltk import FreqDist, word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

### spacy libraries
import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler 
from spacy.tokens import Doc, Span
from spacy.util import filter_spans


import string
from textblob import TextBlob
from stop_words import get_stop_words

# pipelines
from resampling_pipeline import evaluate_method, model_resampling_pipeline, get_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# ml support libraries
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import scipy
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

# ml classifiers
from mlxtend.classifier import EnsembleVoteClassifier
import sklearn
from sklearn.tree import export_graphviz, DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# metrics
import shap
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, \
                            precision_recall_curve, classification_report, \
                            brier_score_loss, log_loss, f1_score, precision_score, \
                            recall_score, make_scorer
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

# Visualization libraries
import chart_studio
import chart_studio.plotly as py
from plotly.subplots import make_subplots
from eval_functions import plot_confusion_matrix, corr_matrix, get_datasets
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import pyLDAvis as vis
import pyLDAvis.gensim
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

import cufflinks as cf
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import seaborn as sns
from eval_functions import plot_confusion_matrix, corr_matrix, get_datasets
from matplotlib import cm
from matplotlib.colors import ListedColormap
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
