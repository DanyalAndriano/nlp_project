#!/usr/bin/env python

import argparse
import datetime
import os
import json
from pathlib import Path
import pandas as pd
import mysql.connector
import preprocessing_functions as pf
from preprocessing_functions import stopwords_list, ext_stopwords_list

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--login', type=str, required=True, help="MySQL login details file path")
parser.add_argument('-a', '--aws', type=str, required=True, help="AWS processed file path")
parser.add_argument('-o', '--out', help="Add output file path")

args = parser.parse_args()

if args.out:
    OUTPUT = Path(args.out)
else:
    CURRENT = Path(os.getcwd())
    OUTPUT = CURRENT/'output_data'
    os.makedirs(OUTPUT, exist_ok=True)
MYSQL_LOGIN = Path(args.login)
AWS_PATH = Path(args.aws)
DATE = datetime.datetime.today().strftime('%Y-%m-%d')

def preprocess_categories(connection, reviews_df, output, date, path):
    df = (pd.read_sql("""
                SELECT rhrc.review_id,
                    GROUP_CONCAT(CASE WHEN rhrc.is_positive = 1
                    THEN rhrc.review_category_id END separator ', ') as category_pos,
                    GROUP_CONCAT(CASE WHEN rhrc.is_positive = 1
                    THEN rc.name END separator ', ') as category_pos_names,
                    GROUP_CONCAT(CASE WHEN rhrc.is_positive = 0
                    THEN rhrc.review_category_id END separator ', ') as category_neg,
                    GROUP_CONCAT(CASE WHEN rhrc.is_positive = 0
                    THEN rc.name END separator ', ') as category_neg_names

                FROM review_has_review_category rhrc
                LEFT JOIN review_category rc
                ON rhrc.review_category_id = rc.id
                GROUP BY review_id;
                """, con=connection)
            .pipe(pf.remove_errors_from_dataframe, output, date)
            .pipe(pf.remove_no_text, output, date)
            .pipe(pf.clean_replace_nulls)
            .pipe(pf.convert_cols)
            .pipe(pf.get_cat_labels)
            .pipe(pf.add_processed, reviews_df))
    return df

def preprocess_reviews(connection, output, date, path):
    df = (pd.read_sql("""
                SELECT IFNULL(r.id, 0) AS review_id,
                IFNULL(r.review_comment_heading, "") AS review_heading,
                IFNULL(r.review_comment, "") AS review_comment,
                r.brand_id as brand_id,
                r.store_id as store_id,
                r.date_reviewed as date,
                r.rating_composite AS rating_composite,
                r.platform_id AS platform_id,
                IFNULL(p.name, "") AS platform_name,
                r.rating_1, r.rating_2, r.rating_3, r.rating_4, r.rating_5

                FROM review r JOIN platform p ON r.platform_id = p.id
                LEFT JOIN brand b ON r.brand_id = b.id

                ORDER BY review_id;
                """, con=connection, parse_dates={'date': {'format': '%Y-%m-%d %H:%M:%S'}})
            .pipe(pf.remove_errors, output, date)
            .pipe(pf.clean_replace_nulls)
            .pipe(pf.average_ratings)
            .pipe(pf.separate_reviews, output, date)
            .assign(combined=lambda x: x.review_comment + ' ' + x.review_heading)
            .drop(['review_comment', 'review_heading'], axis=1)
            .pipe(pf.find_floats, ['combined'])
            .pipe(pf.get_english, path, output, date)
            .assign(raw_tokens=lambda x: x.combined.apply(pf.tokenizer),
                    clean_text=lambda x: x.combined.apply(pf.clean_text),
                    lemmed_tokens=lambda x: x.combined.apply(pf.tokenizer, lem=True),
                    tokens_wsw=lambda x: x.combined.apply(pf.tokenizer, lem=True, stop_words=stopwords_list),
                    tokens_wesw=lambda x: x.combined.apply(pf.tokenizer, lem=True, stop_words=ext_stopwords_list),
                    tokens_wesw_stem=lambda x: x.combined.apply(pf.tokenizer, lem=True, stem=True, stop_words=ext_stopwords_list),
                    word_count=lambda x: x.clean_text.apply(pf.word_count),
                    unique_words=lambda x: x.clean_text.apply(pf.unique_words),
                    char_count=lambda x: x.clean_text.apply(pf.char_count))
            .pipe(pf.remove_erroneous_reviews))
    return df

def preprocess_category_info(connection):
    df = (pd.read_sql("""
                SELECT *
                FROM review_category;
                """, con=connection))
    return df

def preprocess_locations(connection):
    df = (pd.read_sql("""
                SELECT longitude, latitude, country, city, brand_id,
                    store_id, address_line_1, address_line_2
                FROM address
                WHERE longitude
                IS NOT NULL
                GROUP BY store_id;
                """, con=connection)
            .pipe(pf.clean_replace_nulls))
    return df

def main():
    with open(MYSQL_LOGIN, 'r') as f:
        d = json.load(f)
    database = d['database']
    password = d['password']
    CONNECTION = mysql.connector.connect(database=database,
                                         password=password,
                                         host='localhost',
                                         user='root')
    REVIEWS = preprocess_reviews(connection=CONNECTION, output=OUTPUT, date=DATE,
                                 path=AWS_PATH)
    print('Processing categories...')
    CATEGORIES = preprocess_categories(connection=CONNECTION, reviews_df=REVIEWS,
                                       output=OUTPUT, date=DATE, path=AWS_PATH)
    print('Preprocessing locations...')
    LOCATIONS = preprocess_locations(connection=CONNECTION)
    print('Preprocessing category info...')
    CATEGORY_INFO = preprocess_category_info(connection=CONNECTION)

    review_fn = 'reviews-' + DATE + '.json'
    categories_fn = 'categories-' + DATE + '.json'
    location_fn = 'locations-' + DATE + '.json'
    category_info_fn = 'category_info-' + DATE + '.json'

    REVIEWS.to_json(OUTPUT/review_fn, orient='columns', date_format='iso')
    CATEGORIES.to_json(OUTPUT/categories_fn, orient='columns', date_format='iso')
    LOCATIONS.to_json(OUTPUT/location_fn, orient='columns')
    CATEGORY_INFO.to_json(OUTPUT/category_info_fn, orient='columns')
    print('Closing MySQL database connection.')
    CONNECTION.close()
    return REVIEWS, CATEGORIES, LOCATIONS, CATEGORY_INFO

if __name__ == '__main__':
    reviews, categories, locations, category_info = main()
