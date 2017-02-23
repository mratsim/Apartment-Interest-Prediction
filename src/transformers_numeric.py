from scipy.stats import boxcox
import numpy as np
import pandas as pd

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