from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder() 
    le.fit(list(train[sColumn].values) + list(test[sColumn].values))
    
    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn])
        # Warning, there is a bug(?) if trying to us pd.Series(encoded, dtype='category') here
        # mlogloss is negatively impacted by 0,003
        df['encoded_' + sColumn] = encoded
        # CRITICAL, LightGBM coredumps on categorical
        # df['encoded_' + sColumn] = df['encoded_' + sColumn].astype('category')
        return df
    return _trans(train, sColumn, le),_trans(test, sColumn, le)
       

def tr_enc_dispAddr(train, test, y, cache_file):
    sCol = 'display_address'
    trn, tst = _encode_categoricals(train,test, sCol)
    print('Check if detected as categorical: encoded_' + sCol, '  -  ',hasattr(trn['encoded_' + sCol], 'cat'))
    return trn, tst, y, cache_file
    
def tr_enc_manager(train, test, y, cache_file):
    sCol = 'manager_id'
    trn, tst = _encode_categoricals(train,test, sCol)
    print('Check if detected as categorical: encoded_' + sCol, '  -  ',hasattr(trn['encoded_' + sCol], 'cat'))
    return trn, tst, y, cache_file
    
def tr_enc_building(train, test, y, cache_file):
    sCol = 'building_id'
    trn, tst = _encode_categoricals(train,test, sCol)
    print('Check if detected as categorical: encoded_' + sCol, '  -  ',hasattr(trn['encoded_' + sCol], 'cat'))
    return trn, tst, y, cache_file

def tr_enc_streetAddr(train, test, y, cache_file):
    sCol = 'street_address'
    trn, tst = _encode_categoricals(train,test, sCol)
    print('Check if detected as categorical: encoded_' + sCol, '  -  ',hasattr(trn['encoded_' + sCol], 'cat'))
    return trn, tst, y, cache_file