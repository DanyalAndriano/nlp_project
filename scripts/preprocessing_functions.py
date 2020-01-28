import datetime
import itertools
import json
import os
import pathlib
import re
import sys

import matplotlib
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import spacy
import string
import wordninja

from nltk import FreqDist, word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet, brown, reuters
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from stop_words import get_stop_words

with open("additional_stopwords.txt", "r") as textfile:
    additional_stopwords = textfile.read().split('\n')

stopwords_list = list(string.punctuation)
stopwords_list += ['....', '...', '..', '.....', 'im']
stopwords_list += stopwords.words('english') # nltk
stopwords_list += get_stop_words('en') # stop words
stopwords_list = list(set(stopwords_list))
stopwords_list = [ w for w in stopwords_list if w not in 'not' ]

print('Length of standard stopwords list: {}'.format(len(stopwords_list)))

ext_stopwords_list = stopwords_list
ext_stopwords_list += spacy.lang.en.stop_words.STOP_WORDS # spacy
ext_stopwords_list += additional_stopwords
ext_stopwords_list = ENGLISH_STOP_WORDS.union(stopwords_list) # sklearn stopwords
ext_stopwords_list = list(set(ext_stopwords_list))
ext_stopwords_list = [ w for w in ext_stopwords_list if w not in 'not' ]

print('Length of extended stopwords list: {}'.format(len(ext_stopwords_list)))

with open('vocab_20k.txt', 'r', encoding="utf8") as f:
    extended_vocab_20k = f.read().splitlines()

with open('contractions.txt','r') as f:
    contractions = eval(f.read())

def find_review_errors(df):
    """
    List unique identity key for missing and duplicate reviews,
    as well as out of range ratings.

    params:
    df: Dataframe containing review information queried from MySQL WB.
    """

    ratings = ['rating_1', 'rating_2', 'rating_3',
                   'rating_4', 'rating_5', 'rating_composite']
    errors_ids = []
    missing = df.review_id[df[ratings].isnull().all(1)]
    out_of_range = df.review_id[((df[ratings] > 5) | (df[ratings] < 0)).any(1)]
    duplicates = df.review_id[df['review_id'].duplicated()]
    errors_ids += list(missing) + list(out_of_range) + list(duplicates)
    return list(set(errors_ids))

def remove_errors(df, output, date):
    """Remove duplicate reviews, or reviews with missing/inaccurate ratintgs."""
    print('Dataframe loaded from MySQL WorkBench.')
    print('Length of Dataframe: {}'.format(len(df)))
    error_ids = find_review_errors(df)
    error_df = df[df.review_id.isin(error_ids)]
    df = df[~df.review_id.isin(error_ids)].copy()
    print('Errors removed: {}'.format(len(error_ids)))
    print('Length of Corrected Dataframe: {}'.format(len(df)))
    error_df_fn = 'error_df-' + date + '.json'
    error_df.to_json(output/error_df_fn, orient='columns')
    return df

def remove_errors_from_dataframe(df, output, date):
    """Use the errors found in the reviews dataframe to remove erroneous samples
    from the current dataframes."""
    print('Length of Dataframe: {}'.format(len(df)))
    error_df_fn = 'error_df-'+ date + '.json'
    error_df = pd.read_json(output/error_df_fn, orient='columns')
    error_ids = error_df.review_id.tolist()
    df = df[~df.review_id.isin(error_ids)]
    print('Length of Dataframe after removing errors: {}'.format(len(df)))
    return df

def remove_no_text(df, output, date):
    """Use the no text reviews found in the reviews dataframe from
    the current dataframe."""

    print('Length of Dataframe: {}'.format(len(df)))
    no_text_df_fn = 'no_text_df-' + date + '.json'
    no_text_df = pd.read_json(output/no_text_df_fn, orient='columns')
    no_text_ids = no_text_df.review_id.tolist()
    df = df[~df.review_id.isin(no_text_ids)]
    print('Length of Dataframe after removing no text reviews: {}'.format(len(df)))
    return df

def clean_replace_nulls(df):
    """Replace all nulls and empty strings with np.nan (standard numpy null value)."""

    print('Cleaning and replacing nulls...')
    line_endings = ['\r\n', '\r', '\n\n', '\r\r', '\n', '\t', '\t\t']
    df = df.replace(line_endings, ' ', regex=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            for line_ending in line_endings:
                total_line_endings = df[col].str.contains(line_ending, regex=True).sum()
                try:
                    assert total_line_endings == 0
                except:
                    print('The following columns have line endings {}'.format(col))
    list_missing = ['NaN', 'nan', 'None', [None], 'NULL', '', ' ']
    df = df.replace(list_missing, np.nan)
    return df

def convert_cols(df):
    """Return dataframe with columns that sum up negative
    and positive categories for each review."""

    print('Converting Categories...')
    cats = ['category_pos', 'category_neg']
    new_cols = ['pos_count', 'neg_count']
    df = df.copy()
    for cat, new_col in zip(cats, new_cols):
        df[cat] = df[cat].replace(np.nan, '')
        df[cat] = df[cat].astype(str)
        df[cat] = df[cat].apply(lambda x: x.split(',') if x != '' else '')
        df[new_col] = df[cat].apply(lambda x: len(x))
    return df

def get_cat_labels(df):
    """Return sentiment labels based on sum of negative and positive categories
    for each review.

    Sentiment is a multiclass label where: 0 -> negative, 1 -> positive, 2 -> mixed.
    (i.e., was a review positive, negative or mixed.)

    Positive and negative label columns were also addded for multilabel analysis
    (i.e., was a review positive, negative, or both). """

    df = df.copy()
    df.loc[(df['pos_count'] > 0) & (df['neg_count'] == 0), 'sentiment'] = 1
    df.loc[(df['pos_count'] == 0) & (df['neg_count'] > 0), 'sentiment'] = 0
    df.loc[(df['pos_count'] > 0) & (df['neg_count'] > 0), 'sentiment'] = 2

    df.loc[(df['pos_count'] > 0), 'positive'] = 1
    df.loc[(df['neg_count'] > 0), 'negative'] = 1
    df.loc[(df['pos_count'] == 0), 'positive'] = 0
    df.loc[(df['neg_count'] == 0), 'negative'] = 0
    return df

def add_processed(df, reviews_df):
    """Add already preprocessed text and features to current dataframe."""

    print('Adding preprocessed text to categories')
    preprocessed_cut = reviews_df[['review_id','date','combined','clean_text',
                                   'raw_tokens','lemmed_tokens','tokens_wsw',
                                   'tokens_wesw','tokens_wesw_stem', 'word_count',
                                   'unique_words', 'char_count']].copy()
    df = pd.merge(df, preprocessed_cut, on='review_id', how='left')
    df = df.dropna(subset=['combined'])
    return df

def average_ratings(df):
    """ Return dataframe with average rating for reviews with subratings
    (and no composite rating).

    If only the composite exists, fill the new ratings column with
    the previous composite value. """

    ratings = ['rating_1', 'rating_2', 'rating_3',
               'rating_4', 'rating_5', 'rating_composite']

    df['rating'] = df[['rating_1', 'rating_2', 'rating_3',
                       'rating_4', 'rating_5']].mean(axis=1)

    df['rating'] = df['rating'].fillna(df['rating_composite'])
    df['rating'] = df['rating'].apply(lambda x: round(x, 0))

    df = df.drop(ratings, axis=1)
    print('Average ratings calculated.')
    return df

def separate_reviews(df, output, date):
    """Separate review table into text (in either the heading or comment)
    and no-text reviews (no meaningful text, e.g., single digit or letter).

    Save the no text dataframe and return the text dataframe."""

    # Get review ids for text in headings and comments
    print('Separating no text and text reviews...')
    full_length = len(df)
    heading = set(df.review_id[df['review_heading'].notnull()])
    comment = set(df.review_id[df['review_comment'].notnull()])

    # Replace np.nan with empty string to remove floats
    df['review_heading'] = df['review_heading'].replace(np.nan, '')
    df['review_comment'] = df['review_comment'].replace(np.nan, '')

    # Certain reviews contain a single number and should be
    # separated into the no_text reviews dataset
    additional_no_text = []
    for i in df.index:
        if (len(df.review_heading[i]) + len(df.review_comment[i])) <=1:
            additional_no_text.append(df.review_id[i])

    # Create list of review ids for text and no text datasets
    with_text = heading.union(comment)
    with_text -= set(additional_no_text)
    print('Number of samples with text: {}'.format(len(with_text)))

    # Create separate databases with and without review text
    no_text_df = df[~df.review_id.isin(with_text)].copy()
    df = df[df.review_id.isin(with_text)].copy()
    print('Number of text reviews: {}'.format(len(df)))
    print('Number of no text reviews: {}'.format(len(no_text_df)))
    print('Preprocessing Text...')
    no_text_fn = 'no_text_df-' + date + '.json'
    no_text_df.to_json(output/no_text_fn, orient='columns')
    assert len(no_text_df) + len(df) == full_length
    return df

def find_floats(df, text_cols):
    """List all floats in text/object datatype columns."""
    floats = []
    for i in df.index:
        for col in text_cols:
            if isinstance(df[col][i], float):
                floats.append(i)
    print('Number of floats found: {}'.format(len(floats)))
    return df

def get_english(df, path, output, date):
    """Use Amazon Comprehend language detection scores to separate non-English
    and English reviews. Save non-English reviews, return English reviews dataframe."""

    aws_lang_processed = pd.read_json(path, orient='columns')
    full_length = len(df)
    aws_cut = aws_lang_processed[['review_id', 'aws_lang']].copy()
    aws_cut = aws_cut.reset_index(drop=True)
    lang_processed = pd.merge(df, aws_cut, on='review_id', how='left')\
                       .sort_values('aws_lang')\
                       .drop_duplicates('review_id', keep='first')\
                       .dropna(subset=['combined'])\
                       .reset_index(drop=True)
    df = lang_processed[lang_processed.aws_lang == 'en'].copy()
    print('Number of English reviews: {}'.format(len(df)))
    non_english = lang_processed[~lang_processed.index.isin(df.index)].copy()
    print('Number of non-English/unprocessed reviews: {}'.format(len(non_english)))
    non_english_fn = 'non_english-' + date + '.json'
    non_english.to_json(output/non_english_fn, orient='columns')
    return df

def expand_contractions(string_input):
    """Replace contracted words with expanded words.
    Example: can't -> cannot, he'll -> he will.

    contractions.txt with dictionary mappings must be in same directory."""

    string_input = re.sub(r"[^A-Za-z0-9'# ]+", " ", string_input).lower()
    new = re.sub(r"\s+\'", "'", string_input)
    for word in new.split():
        if word in contractions.keys():
            new = new.replace(word, contractions[word])
    return new

def split_hashtag(str_input):
    """" Returns hashtag split into individual words. Wordninja is used
    to split the hashtag after it has been confirmed that it is not a
    single word in the extended english vocabulary (extended_vocab_20k.txt).

    Wordninja probabilistically splits strings."""

    string = re.sub(r"[^A-Za-z0-9# ]+", " ", str_input).lower()
    for word in string.split():
        if word.startswith('#') and word[1:] not in extended_vocab_20k:
            new_words = wordninja.split(word)
            new_words_string = ' '.join(word for word in new_words)
            string = string.replace(word, new_words_string)
    return string

def clean_text(str_input):
    """Returns strings with urls, contractions, new line characters,
    and punctuation removed.  Hashtags are split."""

    no_url = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str_input)
    no_contractions = expand_contractions(no_url)
    split_hash = split_hashtag(no_contractions)
    no_newline = re.sub(r"(?<=[a-z])[\r\n]+"," ", split_hash)
    no_punc = re.sub(r"[^A-Za-z0-9]+", " ", no_newline)
    return no_punc

def tokenizer(str_input, stop_words=None, lem=False, stem=False):
    """ Returns tokenized string. Optional stopwords removal, lemmatization
    or stemming."""

    str_input = clean_text(str_input)
    words = filter(None, str_input.split())
    words = [w.lower() for w in words]
    stemmer = SnowballStemmer("english")
    lemmer = WordNetLemmatizer()
    if stop_words:
        words = [w for w in words if w not in stop_words]
    if lem:
        words = [lemmer.lemmatize(w, "v") for w in words]
    if stem:
        words = [stemmer.stem(w) for w in words]
        if stop_words:
            words = [w for w in words if w not in stop_words]
    return words

def word_count(str_input):
    """ Return total word count for string."""
    return len(str_input.split())

def unique_words(str_input):
    """ Return unique word count for string."""
    return len(set(str_input.split()))

def char_count(str_input):
    """ Return character count for string."""
    return len(str_input)

def remove_erroneous_reviews(df):
    """ Remove reviews with a word count == 0 or a character
    count < 2. Example, single numbers or letters."""
    
    print('Removing any additional erroneous samples.')
    print('Length Dataframe: {}'.format(len(df)))
    remove = df.review_id[(df.word_count == 0) | (df.char_count < 2)]
    print('{} additional non-english or non-alpha numeric samples'.format(len(remove)))
    df = df[~df.review_id.isin(remove)].copy()
    print('Final Length: {}'.format(len(df)))
    return df
