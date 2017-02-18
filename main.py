
# coding: utf-8

# # Import libraries
# * Numerical libraries
# * ScikitLearn Tools
# * Classifier: XGBoost, using the Scikit Learn API
# * time: to name the output files
# * Sklearn-pandas to transform from dataframes to numpy array

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer, LabelBinarizer,RobustScaler,StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# In[3]:

# Impact sklearn_pandas which Pandas DataFrame compatibility with Scikit's classifiers and Pipeline
from sklearn_pandas import DataFrameMapper


# In[4]:

import lightgbm as lgb


# In[5]:

from bs4 import BeautifulSoup 


# In[6]:

import time


# Set random seed for reproducibility

# In[ ]:

np.random.seed(0)


# # Helper functions for functional programming

# In[7]:

# inspired by: https://joshbohde.com/blog/functional-python
# Transformations do not take extra arguments so no need for partial or starmap

from functools import reduce

#compose list of functions (chained composition)
def compose(*funcs):
    def _compose(f, g):
        # functions are expecting X,y not (X,y) so must unpack with *g
        return lambda *args, **kwargs: f(*g(*args, **kwargs))
    return reduce(_compose, funcs)

# pipe function, reverse the order so that it's usual FIFO function order
def pipe(*funcs):
    return compose(*reversed(funcs))


# # Metric: Multiclass logloss

# In[8]:

# Metric Multiclass log loss
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


# # Import and display data

# In[9]:

df_train = pd.read_json(open("./data/train.json", "r"))
df_train.head()


# In[10]:

df_test = pd.read_json(open("./data/test.json", "r"))
df_test.head()


# # Feature Engineering

# In[11]:

# This transformer extracts the number of photos
def transformer_numphot(train, test, y):
    def _trans(df):
        return df.assign(NumPhotos = df['photos'].str.len())
    return _trans(train), _trans(test), y
    
# This transformer extracts the number of features
def transformer_numfeat(train, test, y):
    def _trans(df):
        return df.assign(NumFeat = df['features'].str.len())
    return _trans(train), _trans(test), y
    
# This transformer extracts the number of words in the description
def transformer_numdescwords(train, test, y):
    def _trans(df):
        return df.assign(
            NumDescWords = df["description"].apply(lambda x: len(x.split(" ")))
            )
    return _trans(train), _trans(test), y
    
# This transformer extracts the date/month/year and timestamp in a neat package
def transformer_datetime(train,test, y):
    def _trans(df):
        df = df.assign(
            Created_TS = pd.to_datetime(df["created"])
        )
        return df.assign(
            Created_Year = df["Created_TS"].dt.year,
            Created_Month = df["Created_TS"].dt.month,
            Created_Day = df["Created_TS"].dt.day
            )
    return _trans(train), _trans(test), y

# Bucket nombre de chambres et de bathrooms
# Heure de la journée
# Jour de la semaine
# Retirer numéro de rue
# Imputer les rues sans géoloc
# Quartier (centre le plus proche)
# Distance par rapport au centre
# Clusteriser la latitude/longitude
# manager skill (2*high + medium)

####### Debug Transformer ###########
# Use this transformer anywhere in your Pipeline to dump your dataframe to CSV
def transformer_debug(train,test, y):
    X.to_csv('./debug_train.csv')
    y.to_csv('./debug_test.csv')
    return train,test, y


# ## NLP  on description
# 
# 
# To be refactored, massive leakage due to fold collision

# In[12]:

def transformer_desc_tfidf(train, test, y_train):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
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

    
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_lsa, y_train, test_size=0.2, random_state=42)

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
    print('The mlogloss of prediction is:', multiclass_log_loss(y_test, y_pred))
    
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


    return train_out, test_out, y_train


# # Command Center

# In[13]:

# Feature engineering - sequence of transformations
featurize = pipe(
    transformer_numphot,
    transformer_numfeat,
    transformer_numdescwords,
    transformer_datetime,
    transformer_desc_tfidf
)

# Feature selection - features to keep
mapper = DataFrameMapper([
    (["bathrooms"],RobustScaler()), #Some bathrooms number are 1.5, Some outliers are 112 or 20    (["bedrooms"],OneHotEncoder()),
    (["bedrooms"],RobustScaler()),
    (["latitude"],None),
    (["longitude"],None),
    (["price"],RobustScaler()),
    # (["NumDescWords"],None),
    (["NumFeat"],StandardScaler()),
    (["Created_Year"],None),
    (["Created_Month"],None),
    (["Created_Day"],None),
    (["tfidf_high"],None),
    #(["tfidf_medium"],None),
    #(["tfidf_low"],None)
])


# # Helper functions
# 
# 
# ## Get features that contributes most to the score
# 
# 
# ## Predict and format the output

# In[14]:

####### Predict and format output #######
def output():
    predictions = gbm.predict(np_test, num_iteration=gbm.best_iteration)
    
    #debug
    print(le.classes_)
    print(predictions)
    
    result = pd.DataFrame({
        'listing_id': df_test['listing_id'],
        le.classes_[0]: [row[0] for row in predictions], 
        le.classes_[1]: [row[1] for row in predictions],
        le.classes_[2]: [row[2] for row in predictions]
        })
    result.to_csv('./out/'+time.strftime("%Y-%m-%d_%H%M-")+'-002-feat-eng.csv', index=False)


# # Apply Preproc
#    

# In[15]:

################ Model Selection ################################
y = df_train['interest_level']

df_train, df_test, y_throwaway = featurize(df_train,df_test, y)


X = df_train




# In[16]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:

mapper.fit(X)

X_train = mapper.transform(X_train)

X_test = mapper.transform(X_test)

np_test = mapper.transform(df_test)


# In[18]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(y)

y_train = le.transform(y_train)
y_test = le.transform(y_test)


# In[19]:

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[20]:

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
                num_boost_round=999,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
               feature_name='auto',
               categorical_feature='auto')



# In[21]:

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The mlogloss of prediction is:', multiclass_log_loss(y_test, y_pred))


# In[22]:

######### Most influential features ########
# top_feat()


# In[23]:

######## Predict ########
output()

