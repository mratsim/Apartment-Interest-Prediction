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
                    verbose_eval=False,
                    feature_name='auto',
                    categorical_feature='auto')

    print('Start validating...')
    # predict
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    # eval
    metric = mlogloss(y_val, y_pred)
    print('We stopped at boosting round: ', gbm.best_iteration)
    print('The mlogloss of prediction is: ', metric)
    return gbm, metric