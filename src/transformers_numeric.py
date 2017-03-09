from scipy.stats import boxcox
import numpy as np
import pandas as pd
from math import modf
from sklearn.preprocessing import binarize

# This transformer extracts the number of photos
def tr_numphot(train, test, y, cache_file):
    def _trans(df):
        return df.assign(NumPhotos = df['photos'].str.len())
    return _trans(train), _trans(test), y, cache_file
    
# This transformer extracts the number of features
def tr_numfeat(train, test, y, cache_file):
    def _trans(df):
        return df.assign(NumFeat = df['features'].str.len())
    return _trans(train), _trans(test), y, cache_file
    
# This transformer extracts the number of words in the description
def tr_numdescwords(train, test, y, cache_file):
    def _trans(df):
        return df.assign(
            NumDescWords = df["description"].apply(lambda x: len(x.split(" ")))
            )
    return _trans(train), _trans(test), y, cache_file

#
def tr_log_price(train, test, y, cache_file):
    def _trans(df):
        return df.assign(
            log_price = np.log(df['price'])
            )
    return _trans(train), _trans(test), y, cache_file

def tr_bin_price(train, test, y, cache_file):
    idx_train = train.shape[0]
    train_test = pd.concat((train, test), axis=0)
    
    train_test['Bin_price'] = pd.qcut(train_test['price'], 100, labels=False,duplicates='drop')
    
    trn = train_test.iloc[:idx_train, :]
    tst = train_test.iloc[idx_train:, :]
    return trn, tst, y, cache_file

# Bucket bath and bedroom
def tr_bucket_rooms(train, test, y, cache_file):
    def _trans(df):
        bath = [0,1,2,np.inf]
        bed = [0,1,2,3,np.inf]
        return df.assign(
            bucket_bath = pd.cut(df['bathrooms'], bins=bath,labels=False, include_lowest=True),
            bucket_bed = pd.cut(df['bedrooms'], bins=bed,labels=False, include_lowest=True)
        )
    return _trans(train), _trans(test), y, cache_file

def tr_price_per_room(train, test, y, cache_file): #Assuming always 1 living room
    def _trans(df):
        return df.assign(
            price_per_room = df['price'] / (df['bedrooms'] + 1), # +1 for living room
            price_per_bath = df['price'] / (df['bathrooms'] + 1),
            rooms_sum = df['bedrooms'] + df['bathrooms'],
            rooms_diff = df['bedrooms'] - df['bathrooms'],
            price_per_totalrooms = df['price'] / (df['bedrooms'] + df['bathrooms'] + 1),
            rooms_ratio = (df['bedrooms'] + 1) / df['bathrooms'],
            beds_perc = df['bedrooms'] / (df['bedrooms'] + df['bathrooms'])
        )
    return _trans(train), _trans(test), y, cache_file

def tr_split_bath_toilets(train, test, y, cache_file):
        def _trans(df):
            toilets_only, bathrooms_only  = zip(*df['bathrooms'].map(modf))
            
            return df.assign(
                bathrooms_only = list(map(np.int,bathrooms_only)),
                toilets_only = np.where(np.array(toilets_only)<0, 1, 0)
            )
        return _trans(train), _trans(test), y, cache_file