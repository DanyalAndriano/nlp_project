# python libraries
import collections
import itertools
import json
import logging
import math
import os
import pathlib
import random
import re
import requests
import sys
import time
import zipfile
from collections import Counter, Iterable
from functools import reduce
from itertools import cycle
from pathlib import Path
from requests.auth import HTTPBasicAuth
from time import time

# general libraries
import collections
import itertools
import json
import mysql.connector
import numpy as np
import pandas as pd
import requests
from collections import Counter
from functools import reduce
from requests.auth import HTTPBasicAuth

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

# ml libraries
import scipy
import scipy.sparse as sp
import sklearn
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from mlxtend.classifier import EnsembleVoteClassifier
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier 
from xgboost import XGBClassifier

# NLP libraries
import gensim
import nltk
import spacy
import string
import wordninja
from gensim import corpora
from gensim.models import Word2Vec, Phrases
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.wrappers import LdaMallet
from nltk.corpus import stopwords, wordnet, brown, reuters 
from nltk import collocations
from nltk import FreqDist, word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
from spacy import displacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler 
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from stop_words import get_stop_words
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# pipelines
from resampling_pipeline import evaluate_method, model_resampling_pipeline, get_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Visualization libraries
import chart_studio
import chart_studio.plotly as py
import cufflinks as cf
import matplotlib.pyplot as plt
import pandas_bokeh
import pyLDAvis as vis
import pyLDAvis.gensim
import plotly
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS

# scripts
import calibration_methods
from calibration_functions import classwise_diagrams, calc_classwise_ece, brier_multi, \
                                  confidence_diagram, reliability_diagrams, eval_metrics
from calibration_methods import TemperatureScaling, DirichletScaling, Uncalibrated, Isotronic
from eval_functions import corr_matrix, get_datasets, multiclass_roc_curve, plot_confusion_matrix,\
                           get_ratios_multiclass, plot_cm_best_estimator, plot_ngrams, parse_results


