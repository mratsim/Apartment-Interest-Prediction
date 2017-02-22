# Classifier
import lightgbm as lgb
import xgboost as xgb
from src.metrics import mlogloss

def training_step(x_trn, x_val, y_trn, y_val):
    
    clf, y_pred = _clf_xgb(x_trn, x_val, y_trn, y_val)
    # clf, y_pred = _clf_lgb(x_trn, x_val, y_trn, y_val)

    # eval
    metric = mlogloss(y_val, y_pred)
    # print('We stopped at boosting round: ', clf.best_iteration)
    print('We stopped at boosting round: ', clf.best_ntree_limit)
    print('The mlogloss of prediction is: ', metric)
    return clf, metric


def _clf_lgb(x_trn, x_val, y_trn, y_val):
    # print('Type of data - check especially for categorical')
    # print(x_trn.dtypes)
    
    lgb_train = lgb.Dataset(x_trn, y_trn)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': {'multi_logloss'},
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        #'bagging_fraction': 0.8,
        'max_depth': 6,
        'bagging_freq': 0,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2048,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=True,
                    feature_name='auto')
    
    print('Start validating...')
    # predict
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    return gbm, y_pred

def _clf_xgb(x_trn, x_val, y_trn, y_val, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(x_trn, label=y_trn)
    xgtest = xgb.DMatrix(x_val, label=y_val)
    
    print('Start training...')
    # train
    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    print('Start validating...')
    # predict
    y_pred = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    return model, y_pred