import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split

# Classifier
import lightgbm as lgb

# Clean up text
from bs4 import BeautifulSoup

# Local helper function
from src.pipe import pipe
from src.metrics import mlogloss


# Massive leakage, check cross val predict
def tr_tfidf_lsa(train, test, y):
    vectorizer = TfidfVectorizer(max_features=2**16,
                             min_df=2, stop_words='english',
                             use_idf=True)
    def _preproc(df):
        def _toBeautifulText(text):
            bs =BeautifulSoup(text, "html.parser")
            for br in bs.find_all("br"):
                br.replace_with(" ")
            return bs.get_text()

        return df.assign(
                    RawText = df["description"].apply(lambda x: _toBeautifulText(x))
                    )
    
    train_raw = _preproc(train)['RawText']
    train_vect = vectorizer.fit_transform(train_raw)
    
    test_raw = _preproc(test)['RawText']
    test_vect = vectorizer.transform(test_raw)
    # print(vectorizer.get_feature_names())
    
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))

    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(train_vect)
    X_test_lsa = lsa.transform(test_vect)
    #explained_variance = svd.explained_variance_ratio_.sum()
    #print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_lsa, y, test_size=0.2, random_state=42)

    le = LabelEncoder()
    le.fit(y)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': {'multi_logloss'},
        'learning_rate': 0.1,
        #'feature_fraction': 0.9,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training TF-IDF...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=999,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                   feature_name='auto',
                   categorical_feature='auto')
    
    print('Start validating TF-IDF...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The mlogloss of prediction is:', mlogloss(y_test, y_pred))
    
    print('Start predicting TF-IDF...')
    train_predictions = gbm.predict(X_train_lsa, num_iteration=gbm.best_iteration)
    test_predictions = gbm.predict(X_test_lsa, num_iteration=gbm.best_iteration)

    tfidf_train_names = {
        'tfidf_' + le.classes_[0]: [row[0] for row in train_predictions],
        'tfidf_' + le.classes_[1]: [row[1] for row in train_predictions],
        'tfidf_' + le.classes_[2]: [row[2] for row in train_predictions]
    }
    
    tfidf_test_names = {
        'tfidf_' + le.classes_[0]: [row[0] for row in test_predictions],
        'tfidf_' + le.classes_[1]: [row[1] for row in test_predictions],
        'tfidf_' + le.classes_[2]: [row[2] for row in test_predictions]
    }
    
    train_out = train.assign(**tfidf_train_names)
    test_out = test.assign(**tfidf_test_names)


    return train_out, test_out, y