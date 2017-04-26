import pandas as pd
import numpy as np
import random

def tr_manager_magic(train, test, y, folds, cache_file):
    index=list(range(train.shape[0]))
    # Set random seed or use train_test_split
    random.shuffle(index)
    
    
    train['manager_magic_low'] = np.nan
    train['manager_magic_medium'] = np.nan
    train['manager_magic_high'] = np.nan

    for i in range(5):
        test_index = index[int((i*train.shape[0])/5):int(((i+1)*train.shape[0])/5)]
        train_index = list(set(index).difference(test_index)) 

        cv_train = train.iloc[train_index]
        cv_test  = train.iloc[test_index]

        for m in cv_train.groupby('manager_id'):
            test_subset = cv_test[cv_test.manager_id == m[0]].index

            train.loc[test_subset, 'manager_magic_low'] = (m[1].interest_level == 'low').mean()
            train.loc[test_subset, 'manager_magic_medium'] = (m[1].interest_level == 'medium').mean()
            train.loc[test_subset, 'manager_magic_high'] = (m[1].interest_level == 'high').mean()

    # Test data
    test['manager_magic_low'] = np.nan
    test['manager_magic_medium'] = np.nan
    test['manager_magic_high'] = np.nan

    for m in train.groupby('manager_id'):
        test_subset = test[test.manager_id == m[0]].index

        test.loc[test_subset, 'manager_magic_low'] = (m[1].interest_level == 'low').mean()
        test.loc[test_subset, 'manager_magic_medium'] = (m[1].interest_level == 'medium').mean()
        test.loc[test_subset, 'manager_magic_high'] = (m[1].interest_level == 'high').mean()
    
    return train, test, y, folds, cache_file

def tr_building_magic(train, test, y, folds, cache_file):
    index=list(range(train.shape[0]))
    # Set random seed or use train_test_split
    random.shuffle(index)
    
    
    train['building_magic_low'] = np.nan
    train['building_magic_medium'] = np.nan
    train['building_magic_high'] = np.nan

    for i in range(5):
        test_index = index[int((i*train.shape[0])/5):int(((i+1)*train.shape[0])/5)]
        train_index = list(set(index).difference(test_index)) 

        cv_train = train.iloc[train_index]
        cv_test  = train.iloc[test_index]

        for m in cv_train.groupby('building_id'):
            test_subset = cv_test[cv_test.manager_id == m[0]].index

            train.loc[test_subset, 'building_magic_low'] = (m[1].interest_level == 'low').mean()
            train.loc[test_subset, 'building_magic_medium'] = (m[1].interest_level == 'medium').mean()
            train.loc[test_subset, 'building_magic_high'] = (m[1].interest_level == 'high').mean()

    # Test data
    test['building_magic_low'] = np.nan
    test['building_magic_medium'] = np.nan
    test['building_magic_high'] = np.nan

    for m in train.groupby('building_id'):
        test_subset = test[test.manager_id == m[0]].index

        test.loc[test_subset, 'building_magic_low'] = (m[1].interest_level == 'low').mean()
        test.loc[test_subset, 'building_magic_medium'] = (m[1].interest_level == 'medium').mean()
        test.loc[test_subset, 'building_magic_high'] = (m[1].interest_level == 'high').mean()
    
    return train, test, y, folds, cache_file