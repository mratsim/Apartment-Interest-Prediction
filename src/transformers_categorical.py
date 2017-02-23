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
    
    df_mngr = (pd.concat([train['manager_id'], 
                           pd.get_dummies(train['interest_level'])], axis = 1)
                                        .groupby('manager_id')
                                        .mean()
                                        .rename(columns = lambda x: 'mngr_percent_' + x)
                                           )
    df_mngr['mngr_count']=train.groupby('manager_id').size()
    df_mngr['mngr_skill'] = df_mngr['mngr_percent_high']*2 + df_mngr['mngr_percent_medium']
    # get ixes for unranked managers...
    unrkd_mngrs_ixes = df_mngr['mngr_count']<20
    # ... and ranked ones
    rkd_mngrs_ixes = ~unrkd_mngrs_ixes

    # compute mean values from ranked managers and assign them to unranked ones
    mean_val = df_mngr.loc[rkd_mngrs_ixes,
                           ['mngr_percent_high',
                            'mngr_percent_low',
                            'mngr_percent_medium',
                            'mngr_skill']].mean()
    df_mngr.loc[unrkd_mngrs_ixes, ['mngr_percent_high',
                                    'mngr_percent_low',
                                    'mngr_percent_medium',
                                    'mngr_skill']] = mean_val.values

    trn = train.merge(df_mngr.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
    tst = test.merge(df_mngr.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
        
    new_mngr_ixes = tst['mngr_percent_high'].isnull()
    tst.loc[new_mngr_ixes,['mngr_percent_high',
                                    'mngr_percent_low',
                                    'mngr_percent_medium',
                                    'mngr_skill']]  = mean_val.values

    return trn, tst, y, cache_file