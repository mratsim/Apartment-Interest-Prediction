from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# WARNING - Deprecated - LabelEncoder is directly usable at the feat selection stage
# Furthermore LightGBM core dumped on pandas 'category'
# LightGBM removed categoricals support on commit 10212b5 around Feb 18

# Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder() 
    le.fit(list(train[sColumn].values) + list(test[sColumn].values))
    
    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn])
        df['encoded_' + sColumn] = encoded
        # df['encoded_' + sColumn] = df['encoded_' + sColumn].astype('category')
        return df
    return _trans(train, sColumn, le),_trans(test, sColumn, le)
