import pandas as pd
from textblob import TextBlob

# cache
import os.path #Note: it might be safer to use pathlib, to make sure directory/subdirectory context is kept
import shelve
from pickle import HIGHEST_PROTOCOL
from src.cache import load_from_cache, save_to_cache

def tr_sentiment(train, test, y, folds, cache_file):
    
    print("############# Sentiment Analysis step ################")
    cache_key_train = 'nlp_sentiment_train'
    cache_key_test = 'nlp_sentiment_test'
    
    #Check if cache file exist and if data for this step is cached
    dict_train, dict_test = load_from_cache(cache_file, cache_key_train, cache_key_test)
    if dict_train is not None and dict_test is not None:
        train_out = train.assign(**dict_train)
        test_out = test.assign(**dict_test)
        return train_out, test_out, y, folds, cache_file

    print('# No cache detected, computing from scratch #')

    def _trans(df):
        return {
            'sentiment_polarity': df['description'].apply(lambda x: TextBlob(x).sentiment.polarity),
            'sentiment_subjectivity': df['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        }
    
    nlp_sentiment_train = _trans(train)
    nlp_sentiment_test = _trans(test)
        
    print('Caching features in ' + cache_file)
    save_to_cache(cache_file, cache_key_train, cache_key_test, nlp_sentiment_train, nlp_sentiment_test)
    
    print('Adding features to dataframe')
    train_out = train.assign(**nlp_sentiment_train)
    test_out = test.assign(**nlp_sentiment_test)

    return train_out,test_out, y, folds, cache_file