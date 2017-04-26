from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import numpy as np

# According to Sklearn core devs, LabelEncoder should only be used for label (y).
# It does not provide the same interface as others as it does not take X, y parameter, only y
# As such it cannot be use in a Pipeline.

def tr_lower_address(train, test, y, folds, cache_file):
    def _trans(df):
        return df.assign(
            lower_disp_addr = df['display_address'].apply(str.lower),
            lower_street_addr = df['street_address'].apply(str.lower)
        )
    return _trans(train),_trans(test), y, folds, cache_file

# DEPRECATED Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder() 
    le.fit(list(train[sColumn].apply(str.lower).values) + list(test[sColumn].apply(str.lower).values))
    
    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn].apply(str.lower))
        df['encoded_' + sColumn] = encoded
        # df['encoded_' + sColumn] = df['encoded_' + sColumn].astype('category')
        return df
    return _trans(train, sColumn, le),_trans(test, sColumn, le)

def tr_encoded_manager(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train,test,"manager_id")
    return trn, tst, y, folds, cache_file
def tr_encoded_building(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train,test,"building_id")
    return trn, tst, y, folds, cache_file
def tr_encoded_disp_addr(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train,test,"display_address")
    return trn, tst, y, folds, cache_file
def tr_encoded_street_addr(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train,test,"street_address")
    return trn, tst, y, folds, cache_file

def tr_filtered_display_addr(train, test, y, folds, cache_file):
    address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
    }
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    def _address_map_func(s):
        s = s.split(' ')
        out = []
        for x in s:
            if x in address_map:
                out.append(address_map[x])
            else:
                out.append(x)
        return ' '.join(out)
    def _trans(df):
        df = df.assign(
            filtered_address = df['display_address']
                                    .apply(str.lower)
                                    .apply(lambda x: x.translate(remove_punct_map))
                                    .apply(lambda x: _address_map_func(x))
        )
        new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

        for col in new_cols:
            df[col] = df['filtered_address'].apply(lambda x: 1 if col in x else 0)

        df['other_address'] = df[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
        return df
    
    return _trans(train), _trans(test), y, folds, cache_file

def tr_dedup_features(train, test, y, folds, cache_file):
    def _clean(s):
        x = s.replace("-", "")
        x = x.replace(" ", "")
        x = x.replace("twenty four hour", "24")
        x = x.replace("24/7", "24")
        x = x.replace("24hr", "24")
        x = x.replace("24-hour", "24")
        x = x.replace("24hour", "24")
        x = x.replace("24 hour", "24")
        x = x.replace("common", "cm")
        x = x.replace("concierge", "doorman")
        x = x.replace("housekeep", "doorman")
        x = x.replace("in_super", "doorman")
        x = x.replace("bicycle", "bike")
        x = x.replace("private", "pv")
        x = x.replace("deco", "dc")
        x = x.replace("decorative", "dc")
        x = x.replace("onsite", "os")
        x = x.replace("outdoor", "od")
        x = x.replace("dogs", "dog")
        x = x.replace("cats", "cat")
        x = x.replace("no fee", "nofee")
        x = x.replace("no-fee", "nofee")
        x = x.replace("no_fee", "nofee")
        x = x.replace("reduced_fee", "lowfee")
        x = x.replace("reduced fee", "lowfee")
        x = x.replace("low_fee", "lowfee")
        x = x.replace("Exclusive", "exclusive")
        x = x.replace("pre_war", "prewar")
        x = x.replace("pre war", "prewar")
        x = x.replace("pre-war", "prewar")
        x = x.replace("lndry", "laundry")
        x = x.replace("Laundry in Unit", "laundry")
        x = x.replace("gym", "health")
        x = x.replace("fitness", "health")
        x = x.replace("training", "health")
        x = x.replace("train", "transport")
        x = x.replace("subway", "transport")
        x = x.replace("heat water", "utilities")
        x = x.replace("water included", "utilities")
        x = x.replace("Dishwasher", "utilities")
        x = x.replace("Elevator", "elevator")
        return x
    def _encode_features(l):
        k = 4
        return list(set(map(lambda x: _clean(x)[:k].strip(), l))) #convert to set first to remove duplicate then back to list
    def _trans(df):
        return df.assign(
            dedup_features = df['features'].apply(_encode_features).str.join(", ")
        )
    return _trans(train), _trans(test), y, folds, cache_file
    


#############
# Bins managers and building
# Since we don't use y, there shouldn't be leakage if we use the total count train + test.
# Issue: the current version can put 2 same  building in separate bins
def tr_bin_buildings_mngr(train, test, y, folds, cache_file):
    def _trans(df):
        return df.assign(
            #duplicates = drop avoids error whena single value would need to appear in 2 different bins
            #It needs pandas version>=20.0
            #It's probably better to cut by value if distrbution between training and tests are different
            
            Bin_Buildings = pd.qcut(df['building_id'].value_counts(), 30, labels=False,duplicates='drop'),
            Bin_Managers = pd.qcut(df['manager_id'].value_counts(), 30, labels=False, duplicates='drop')
            )
    return _trans(train), _trans(test), y, folds, cache_file

##############
# V2 version which encode the manager/building in a new column
# Slight leak but probably fine since semi-supervised learning/don't use labels
def tr_bin_buildings_mngr_v2(train, test, y, folds, cache_file):
    def _check_percentile(series_valcount,percentile):
        cutoff = 100 - percentile
        return series_valcount.index.values[
                        series_valcount.values >= np.percentile(series_valcount.values, cutoff)
              ]
    
    mngr_count = pd.concat((train['manager_id'], test['manager_id']), axis=0).value_counts()
    bdng_count = pd.concat((train['building_id'], test['building_id']), axis=0).value_counts()
    
    def _trans(df, mngr_count, bdng_count):
        dict_top_mngr = {'top_' + str(p) + '_manager':
                         df['manager_id'].isin(_check_percentile(mngr_count,p)).astype(np.int)
                         for p in [1,2,5,10,15,20,25,30,50]}
        dict_top_bdng = {'top_' + str(p) + '_building':
                         df['building_id'].isin(_check_percentile(bdng_count,p)).astype(np.int)
                         for p in [1,2,5,10,15,20,25,30,50]}
        
        return df.assign(**dict_top_mngr,**dict_top_bdng)
    
    trn = _trans(train,mngr_count,bdng_count)
    tst = _trans(test,mngr_count,bdng_count)
    
    return trn, tst, y, folds, cache_file
    