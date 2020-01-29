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
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import interp
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.decomposition import PCA  

# calibration
import calibration
import calibration.binning as binning
import calibration.stats as stats
import netcal
from netcal.binning import IsotonicRegression
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from scipy.optimize import minimize 
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# fast bert dependencies
from box import Box
import fast_bert
import torch
import torchvision
from fast_bert.data_cls import BertDataBunch
from fast_bert.data_lm import BertLMDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.learner_lm import BertLMLearner
from fast_bert.metrics import fbeta, F1, accuracy_multilabel, accuracy_thresh, accuracy
from fast_bert.prediction import BertClassificationPredictor

# geospatial libraries
import geopandas as gpd
import geopandas.tools
import plotly.express as px
import plotly.graph_objects as go
from pygeocoder import Geocoder
from shapely.geometry import Point, Polygon

# metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, \
                            precision_recall_curve, classification_report, \
                            brier_score_loss, log_loss, f1_score, precision_score, \
                            recall_score, make_scorer, mean_squared_error, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

# ml classifiers
import xgboost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier 
from xgboost import XGBClassifier

# nlp libraries
import en_core_web_lg
import gensim
import nltk
import spacy
from gensim import corpora
from gensim.models import Word2Vec, Phrases
from nltk import FreqDist, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy import displacy
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

# pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# scripts
import calibration_methods
from calibration_functions import classwise_diagrams, calc_classwise_ece, brier_multi, \
                                  confidence_diagram, reliability_diagrams, eval_metrics
from calibration_methods import TemperatureScaling, DirichletScaling, Uncalibrated, Isotronic
from eval_functions import corr_matrix, get_datasets, multiclass_roc_curve, plot_confusion_matrix,\
                           get_ratios_multiclass, plot_cm_best_estimator, plot_ngrams, parse_results

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap