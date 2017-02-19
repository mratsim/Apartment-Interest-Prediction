from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split

# Classifier
from lightgbm import LGBMClassifier

# Clean up text
from bs4 import BeautifulSoup

# Local helper function
from src.pipe import pipe
from src.metrics import mlogloss
from src.oof_predict import out_of_fold_predict

from sklearn.model_selection import KFold

# cache
import os.path #Note: it might be safer to use pathlib, to make sure directory/subdirectory context is kept
import shelve
from pickle import HIGHEST_PROTOCOL
from src.cache import load_from_cache, save_to_cache

# Massive leakage, check cross val predict
def tr_tfidf_lsa_lgb(train, test, y, cache_file):
    print("############# TF-IDF + LSA step ################")
    cache_key_train = 'tfidf_lsa_lgb_train'
    cache_key_test = 'tfidf_lsa_lgb_test'
    
    #Check if cache file exist and if data for this step is cached
    dict_train, dict_test = load_from_cache(cache_file, cache_key_train, cache_key_test)
    if dict_train is not None and dict_test is not None:
        train_out = train.assign(**dict_train)
        test_out = test.assign(**dict_test)
        return train_out, test_out, y, cache_file

    print('# No cache detected, computing from scratch #')
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

    le = LabelEncoder()
    y_encode = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_lsa, y_encode, test_size=0.2, random_state=42)

    # train
    gbm = LGBMClassifier(
        n_estimators=2048,
        seed=42,
        objective='multiclass',
        colsample_bytree='0.8',
        subsample='0.8'
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1337)

    print('Start training - Number of folds: ', kf.get_n_splits())
    train_predictions = out_of_fold_predict(gbm, X_train_lsa, y_encode, kf.split(X_train_lsa))

    tfidf_train_names = {
        'tfidf_' + le.classes_[0]: [row[0] for row in train_predictions],
        'tfidf_' + le.classes_[1]: [row[1] for row in train_predictions],
        'tfidf_' + le.classes_[2]: [row[2] for row in train_predictions]
    }
    
    gbm.fit(X_train,y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose = False
           )
    
    print('Start validating TF-IDF + LSA...')
    # predict
    y_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('We stopped at boosting round: ', gbm.best_iteration)
    print('The mlogloss of prediction is:', mlogloss(y_test, y_pred))
    
    print('Start predicting TF-IDF + LSA...')
    test_predictions = gbm.predict_proba(X_test_lsa, num_iteration=gbm.best_iteration)
    
    tfidf_test_names = {
        'tfidf_' + le.classes_[0]: [row[0] for row in test_predictions],
        'tfidf_' + le.classes_[1]: [row[1] for row in test_predictions],
        'tfidf_' + le.classes_[2]: [row[2] for row in test_predictions]
    }
    
    print('Caching features in ' + cache_file)
    save_to_cache(cache_file, cache_key_train, cache_key_test, tfidf_train_names, tfidf_test_names)
    
    print('Adding features to dataframe')
    train_out = train.assign(**tfidf_train_names)
    test_out = test.assign(**tfidf_test_names)


    return train_out, test_out, y, cache_file