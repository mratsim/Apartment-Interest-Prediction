### Out of fold predictions for XGBoost and LightGBM with early stopping support
### Do not compute cros_val in parallel as XGB and LGB are already parallel
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.metrics import mlogloss
import numpy as np

def out_of_fold_predict(est, X, y, cv):
    
    # Create an array of 0 with the shape of the result (3 classes = 3 columns)
    result = np.zeros([y.shape[0], 3])
    
    n = 1
    for train_idx, valid_idx in cv:
        print('#################################')
        print('#########  Training for fold: ', n)
        # Make sure there is no estimator leakage by fitting a clone
        clf = clone(est)
        # Get a validation set for early stopping
        x_trn, x_val, y_trn, y_val = train_test_split(X[train_idx], y[train_idx], test_size=0.2, random_state=42)
        clf.fit(x_trn,y_trn,
            eval_set=[(x_val, y_val)],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose=False
           )
        stop_round = clf.best_iteration
        print('#######  Validating for fold: ', n)
        # predict
        y_pred = clf.predict_proba(x_val, num_iteration=stop_round)
        # eval
        print('We stopped at boosting round: ', stop_round)
        print('The mlogloss of prediction is:', mlogloss(y_val, y_pred))
        
        print('#######  Retraining on whole fold: ', n)
        clf = clone(est)
        clf.fit(X[train_idx], y[train_idx], verbose=False)
        
        stop_round = np.int(stop_round*1.1)
        
        # Predict out-of-fold
        print('#######  Predicting for fold: ', n)
        result[valid_idx] = clf.predict_proba(X[valid_idx], num_iteration=stop_round)
        n +=1
    return result
