from sklearn.preprocessing import LabelEncoder
import pandas as pd
# import pandas as pd

# WARNING - Deprecated - LabelEncoder is directly usable at the feat selection stage
# Furthermore LightGBM core dumped on pandas 'category'
# LightGBM removed categoricals support on commit 10212b5 around Feb 18

# Besides, according to Sklearn core devs, LabelEncoder should only be used for label (y).
# It does not provide the same interface as others as it does not take X, y parameter, only y
# As such it cannot be use in a Pipeline.


# DEPRECATED Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder() 
    le.fit(list(train[sColumn].values) + list(test[sColumn].values))
    
    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn])
        df['encoded_' + sColumn] = encoded
        # df['encoded_' + sColumn] = df['encoded_' + sColumn].astype('category')
        return df
    return _trans(train, sColumn, le),_trans(test, sColumn, le)



#############
# Manager skill
def tr_managerskill(train, test, y, cache_file):
    # Beware of not leaking "mean" or frequency from train to test.
    
    df_manager = train['manager_id']
    
    trn = train.join( pd.concat([df_manager, pd.get_dummies(y)], axis = 1)
            .groupby('manager_id')
            .transform('mean')
            .rename(columns = lambda x: 'Percent_manager_' + x)
    )
        
    tst = test.join(trn[['Percent_manager_high',
                         'Percent_manager_low',
                         'Percent_manager_medium']],
                         how='left')

    return trn, tst, y, cache_file