from __future__ import print_function

import boto3
import time
import sys
from io import StringIO
import os
import shutil

import argparse
from collections import Counter
import csv
import json
import numpy as np
import pandas as pd
import re

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from custom import SelectText, SelectSentiment, TextFeatures, clean_text, expand_contractions

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'text', 
    'rating', 
    'aws_neg',
    'aws_pos',
    'aws_mixed'] 

label_column = 'target'
weights_column = 'weights'

feature_dtype = {
    'text': object, 
    'rating': np.int64, 
    'aws_neg': np.float64,
    'aws_pos': np.float64,
    'aws_mixed': np.float64} 

label_dtype = {'target': np.int64} # 0 - neg, 1 - pos, 2 - mix
weights_dtype = {'weights': np.float64} # 0 - neg, 1 - pos, 2 - mix

def merge_dicts(x, y, w=None):
    z = x.copy()   # start with x's keys and values
    z.update(y)
    if w:
        z.update(w)
    return z


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(
        file,
        header=None,
        names=feature_columns_names + [label_column] + [weights_column],
        dtype=merge_dicts(feature_dtype, label_dtype, weights_dtype)) for file in input_files ]
    concat_data = pd.concat(raw_data)
    print(concat_data.head())



#----------------PIPELINE------------------------------------------------------
    select_text = FunctionTransformer(SelectText, validate=False, accept_sparse=False)
    select_sentiments = FunctionTransformer(SelectSentiment, validate=False, accept_sparse=False)
    text_features = FunctionTransformer(TextFeatures, validate=False, accept_sparse=False)
    
    
    tfidf_pipe = Pipeline([
        ('text', select_text),
        ('tfidf', TfidfVectorizer(max_features=70000,ngram_range=(1,4))),
        ('svd', TruncatedSVD(algorithm='randomized', random_state=42, n_components=50))])

    features_pipe = Pipeline([
        ('text', select_text),
        ('features', text_features)])

    sentiment_pipe = Pipeline([
        ('features', select_sentiments)])

    feature_union_pipe = FeatureUnion([
        ('tfidf', tfidf_pipe),
        ('features', features_pipe),
        ('sentiment', sentiment_pipe)])
        
        
    preprocessor = feature_union_pipe.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        #df = pd.read_csv(StringIO(u""+input_data.to_csv(header=None,index=False)), header=None)
        df = pd.read_csv(StringIO(input_data), header=None, encoding='utf-8')
        
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the sentiment label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names) + 2:
            # This is labelled data with class weights
            df.columns = feature_columns_names + [label_column] + [weights_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example with no weights
            df.columns = feature_columns_names

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def model_fn(model_dir):
    """Deserialize fitted model.
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    The output is returned in the following order:
    """
    
    features = model.transform(input_data)
    

    if label_column in input_data.columns and weights_column in input_data.columns:
         # Return the label (as the first column) and weights as second 
            # + the set of features.
        features = np.insert(features, 0, input_data[weights_column], axis=1)
        features = np.insert(features, 0, input_data[label_column], axis=1)
       
        return features
    
    elif label_column in input_data.columns:
        # Return the label (as the first column) + the set of features.
        features = np.insert(features, 0, input_data[label_column], axis=1)
        return features
    
    else:
        # Return only the set of features
        return features

    
def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
