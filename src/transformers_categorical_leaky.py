#DON'T USE, Leaky

#############
# Manager skill
def tr_managerskill(train, test, y, cache_file):
    # Beware of not leaking "mean" or frequency from train to test.
    # WARNING Leak like crazy - TO REFACTOR
    
    df_mngr = (pd.concat([train['manager_id'], 
                           pd.get_dummies(train['interest_level'])], axis = 1)
                                        .groupby('manager_id')
                                        .mean()
                                        .rename(columns = lambda x: 'mngr_percent_' + x)
                                           )
    df_mngr['mngr_count']=train.groupby('manager_id').size()
    df_mngr['mngr_skill'] = df_mngr['mngr_percent_high']*2 + df_mngr['mngr_percent_medium']
    # get ixes for unranked managers...
    unrkd_mngrs_ixes = df_mngr['mngr_count']==1 #<20
    # ... and ranked ones
    rkd_mngrs_ixes = ~unrkd_mngrs_ixes

    # # compute mean values from ranked managers and assign them to unranked ones
    # mean_val = df_mngr.loc[rkd_mngrs_ixes,
    #                        ['mngr_percent_high',
    #                         'mngr_percent_low',
    #                         'mngr_percent_medium',
    #                         'mngr_skill']].mean()
    df_mngr.loc[unrkd_mngrs_ixes, ['mngr_percent_high',
                                    'mngr_percent_low',
                                    'mngr_percent_medium',
                                    'mngr_skill']] = -1 # mean_val.values

    trn = train.merge(df_mngr.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
    tst = test.merge(df_mngr.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
        
    new_mngr_ixes = tst['mngr_percent_high'].isnull()
    tst.loc[new_mngr_ixes,['mngr_percent_high',
                                    'mngr_percent_low',
                                    'mngr_percent_medium',
                                    'mngr_skill']]  = -1 # mean_val.values

    return trn, tst, y, cache_file

#############
# Building hype
def tr_buildinghype(train, test, y, cache_file):
    # Beware of not leaking "mean" or frequency from train to test.
    # WARNING Leak like crazy - TO REFACTOR
    
    df_bdng = (pd.concat([train['building_id'], 
                           pd.get_dummies(train['interest_level'])], axis = 1)
                                        .groupby('building_id')
                                        .mean()
                                        .rename(columns = lambda x: 'bdng_percent_' + x)
                                           )
    df_bdng['bdng_count']=train.groupby('building_id').size()
    df_bdng['bdng_hype'] = df_bdng['bdng_percent_high']*2 + df_bdng['bdng_percent_medium']
    # get ixes for unranked buildings...
    unrkd_bdngs_ixes = df_bdng['bdng_count'] ==1  # <20
    # ... and ranked ones
    rkd_bdngs_ixes = ~unrkd_bdngs_ixes

    # # compute mean values from ranked buildings and assign them to unranked ones
    # mean_val = df_bdng.loc[rkd_bdngs_ixes,
    #                        ['bdng_percent_high',
    #                         'bdng_percent_low',
    #                         'bdng_percent_medium',
    #                         'bdng_hype']].mean()
    df_bdng.loc[unrkd_bdngs_ixes, ['bdng_percent_high',
                                    'bdng_percent_low',
                                    'bdng_percent_medium',
                                    'bdng_hype']] = -1 # mean_val.values

    trn = train.merge(df_bdng.reset_index(),how='left', left_on='building_id', right_on='building_id')
    tst = test.merge(df_bdng.reset_index(),how='left', left_on='building_id', right_on='building_id')
        
    new_bdng_ixes = tst['bdng_percent_high'].isnull()
    tst.loc[new_bdng_ixes,['bdng_percent_high',
                                    'bdng_percent_low',
                                    'bdng_percent_medium',
                                    'bdng_hype']]  = -1 # mean_val.values

    return trn, tst, y, cache_file