# Classifier
import lightgbm as lgb
from src.metrics import mlogloss

def train_lgb(x_trn, x_val, y_trn, y_val):
    lgb_train = lgb.Dataset(x_trn, y_trn)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': {'multi_logloss'},
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2048,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                   feature_name='auto',
                   categorical_feature='auto')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    # eval
    print('\n\nThe mlogloss of prediction is:', mlogloss(y_val, y_pred))
    return gbm