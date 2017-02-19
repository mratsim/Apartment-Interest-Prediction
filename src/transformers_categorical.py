from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder() 
    le.fit(list(train[sColumn].values) + list(test[sColumn].values))
    
    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn])
        new_col = {
            'encoded_' + sColumn: [row for row in encoded]
        }
        return df.assign(**new_col)
    return _trans(train, sColumn, le),_trans(test, sColumn, le)
       

def tr_enc_dispAddr(train, test, y):
    trn, tst = _encode_categoricals(train,test, 'display_address')
    return trn, tst, y
    
def tr_enc_manager(train, test, y):
    trn, tst = _encode_categoricals(train,test, 'manager_id')
    return trn, tst, y
    
def tr_enc_building(train, test, y):
    trn, tst = _encode_categoricals(train,test, 'building_id')
    return trn, tst, y

def tr_enc_streetAddr(train, test, y):
    trn, tst = _encode_categoricals(train,test, 'street_address')
    return trn, tst, y