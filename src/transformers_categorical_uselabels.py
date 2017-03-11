import numpy as np

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split

# Classifier
from lightgbm import LGBMClassifier

# Local helper function
from src.metrics import mlogloss
from src.oof_predict import out_of_fold_predict

# cache
import os.path #Note: it might be safer to use pathlib, to make sure directory/subdirectory context is kept
import shelve
from pickle import HIGHEST_PROTOCOL
from src.cache import load_from_cache, save_to_cache


# Check for leakage in CV
def tr_managerskill(train, test, y, folds, cache_file):
    print("\n\n############# Manager skill step ################")
    cache_key_train = 'managerskill_train'
    cache_key_test = 'managerskill_test'
    
    #Check if cache file exist and if data for this step is cached
    dict_train, dict_test = load_from_cache(cache_file, cache_key_train, cache_key_test)
    if dict_train is not None and dict_test is not None:
        train_out = train.assign(**dict_train)
        test_out = test.assign(**dict_test)
        return train_out, test_out, y, folds, cache_file

    print('# No cache detected, computing from scratch #')
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(list(train['manager_id'].values) + list(test['manager_id'].values))
    
    X_train_mngr = lb.transform(train['manager_id']).astype(np.float32)
    X_test_mngr = lb.transform(test['manager_id']).astype(np.float32)
    
    
    le = LabelEncoder()
    y_encode = le.fit_transform(y)

    
    
    # Separate train in train + validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train_mngr, y_encode, test_size=0.2, random_state=42)

    # train
    gbm = LGBMClassifier(
        n_estimators=2048,
        seed=42,
        objective='multiclass',
        colsample_bytree='0.8',
        subsample='0.8'
    )
    
    # Predict out-of-folds train data
    print('Start training - Number of folds: ', len(folds))
    train_predictions = out_of_fold_predict(gbm, X_train_mngr, y_encode, folds)

    mngr_train_names = {
        'mngr_' + le.classes_[0]: [row[0] for row in train_predictions],
        'mngr_' + le.classes_[1]: [row[1] for row in train_predictions],
        'mngr_' + le.classes_[2]: [row[2] for row in train_predictions],
    }
    mngr_train_names['mngr_skill'] = [2*h+m for (h,m) in zip(mngr_train_names['mngr_high'],mngr_train_names['mngr_medium'])]
    
    gbm.fit(X_train,y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose = False
           )
    
    # Now validate the predict value using the previously split validation set
    print('Start validating Manager skill...')
    # predict
    y_pred = gbm.predict_proba(X_val, num_iteration=gbm.best_iteration)
    # eval
    print('We stopped at boosting round: ', gbm.best_iteration)
    print('The mlogloss of prediction is:', mlogloss(y_val, y_pred))
    
    # Now compute the value for the actual test data using out-of-folds predictions
    print('Start predicting Manager skill...')
    test_predictions = gbm.predict_proba(X_test_mngr, num_iteration=gbm.best_iteration)
    
    mngr_test_names = {
        'mngr_' + le.classes_[0]: [row[0] for row in test_predictions],
        'mngr_' + le.classes_[1]: [row[1] for row in test_predictions],
        'mngr_' + le.classes_[2]: [row[2] for row in test_predictions]
    }
    mngr_test_names['mngr_skill'] = [2*h+m for (h,m) in zip(mngr_test_names['mngr_high'],mngr_test_names['mngr_medium'])]
    
    print('Caching features in ' + cache_file)
    save_to_cache(cache_file, cache_key_train, cache_key_test, mngr_train_names, mngr_test_names)
    
    print('Adding features to dataframe')
    train_out = train.assign(**mngr_train_names)
    test_out = test.assign(**mngr_test_names)

    return train_out, test_out, y, folds, cache_file


# Check for leakage in CV
def tr_buildinghype(train, test, y, folds, cache_file):
    print("\n\n############# Building hype step ################")
    cache_key_train = 'buildinghype_train'
    cache_key_test = 'buildinghype_test'
    
    #Check if cache file exist and if data for this step is cached
    dict_train, dict_test = load_from_cache(cache_file, cache_key_train, cache_key_test)
    if dict_train is not None and dict_test is not None:
        train_out = train.assign(**dict_train)
        test_out = test.assign(**dict_test)
        return train_out, test_out, y, folds, cache_file

    print('# No cache detected, computing from scratch #')
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(list(train['building_id'].values) + list(test['building_id'].values))
    
    X_train_bdng = lb.transform(train['building_id']).astype(np.float32)
    X_test_bdng = lb.transform(test['building_id']).astype(np.float32)
    
    
    le = LabelEncoder()
    y_encode = le.fit_transform(y)

    
    
    # Separate train in train + validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train_bdng, y_encode, test_size=0.2, random_state=42)

    # train
    gbm = LGBMClassifier(
        n_estimators=2048,
        seed=42,
        objective='multiclass',
        colsample_bytree='0.8',
        subsample='0.8'
    )
    
    # Predict out-of-folds train data
    print('Start training - Number of folds: ', len(folds))
    train_predictions = out_of_fold_predict(gbm, X_train_bdng, y_encode, folds)

    bdng_train_names = {
        'bdng_' + le.classes_[0]: [row[0] for row in train_predictions],
        'bdng_' + le.classes_[1]: [row[1] for row in train_predictions],
        'bdng_' + le.classes_[2]: [row[2] for row in train_predictions]
    }
    bdng_train_names['bdng_hype'] = [2*h+m for (h,m) in zip(bdng_train_names['bdng_high'],bdng_train_names['bdng_medium'])]
    
    gbm.fit(X_train,y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose = False
           )
    
    # Now validate the predict value using the previously split validation set
    print('Start validating Manager skill...')
    # predict
    y_pred = gbm.predict_proba(X_val, num_iteration=gbm.best_iteration)
    # eval
    print('We stopped at boosting round: ', gbm.best_iteration)
    print('The mlogloss of prediction is:', mlogloss(y_val, y_pred))
    
    # Now compute the value for the actual test data using out-of-folds predictions
    print('Start predicting Building hype...')
    test_predictions = gbm.predict_proba(X_test_bdng, num_iteration=gbm.best_iteration)
    
    bdng_test_names = {
        'bdng_' + le.classes_[0]: [row[0] for row in test_predictions],
        'bdng_' + le.classes_[1]: [row[1] for row in test_predictions],
        'bdng_' + le.classes_[2]: [row[2] for row in test_predictions]
    }
    bdng_test_names['bdng_hype'] = [2*h+m for (h,m) in zip(bdng_test_names['bdng_high'],bdng_test_names['bdng_medium'])]
    
    print('Caching features in ' + cache_file)
    save_to_cache(cache_file, cache_key_train, cache_key_test, bdng_train_names, bdng_test_names)
    
    print('Adding features to dataframe')
    train_out = train.assign(**bdng_train_names)
    test_out = test.assign(**bdng_test_names)

    return train_out, test_out, y, folds, cache_file