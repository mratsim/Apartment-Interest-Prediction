# coding: utf-8

# Captain obvious
import numpy as np
import pandas as pd

# feature preprocessing
from sklearn.preprocessing import LabelEncoder, Normalizer, LabelBinarizer,RobustScaler,StandardScaler, OneHotEncoder

from sklearn_pandas import DataFrameMapper

# Custom helper functions
from src.pipe import pipe
from main_output import output
from main_train import train_lgb
from src.preprocessing import preprocessing

# Mesure time
import time

# Set random seed for reproducibility
np.random.seed(1337)

# # Import data
df_train = pd.read_json(open("./data/train.json", "r"))
df_test = pd.read_json(open("./data/test.json", "r"))

X = df_train
X_test = df_test
y = df_train['interest_level']
idx_test = df_test['listing_id']

###### TODO ###########
# Bucket nombre de chambres et de bathrooms
# Heure de la journée
# Jour de la semaine
# Retirer numéro de rue
# Imputer les rues sans géoloc
# Quartier (centre le plus proche)
# Distance par rapport au centre
# Clusteriser la latitude/longitude
# manager skill (2*high + medium)

#######################

# # Command Center

from src.transformers_numeric import tr_numphot, tr_numfeat, tr_numdescwords
from src.transformers_time import tr_datetime
#from src.transformers_debug import tr_dumpcsv
from src.transformers_nlp_tfidf import tr_tfidf_lsa

# Feature engineering - sequence of transformations
tr_pipeline = pipe(
    tr_numphot,
    tr_numfeat,
    tr_numdescwords,
    tr_datetime
#    tr_tfidf_lsa
)

# Feature selection - features to keep
select_feat = DataFrameMapper([
    (["bathrooms"],RobustScaler()),
    (["bedrooms"],RobustScaler()),
    (["latitude"],None),
    (["longitude"],None),
    (["price"],RobustScaler()),
    # (["NumDescWords"],None),
    (["NumFeat"],StandardScaler()),
    (["Created_Year"],None),
    (["Created_Month"],None),
    (["Created_Day"],None)
    #(["tfidf_high"],None),
    #(["tfidf_medium"],None),
    #(["tfidf_low"],None)
])

################ Preprocessing #####################
x_trn, x_val, y_trn, y_val, X_test, labelencoder = preprocessing(
   X, X_test, y, tr_pipeline, select_feat)

############ Train and Validate ####################
gbm = train_lgb(x_trn, x_val, y_trn, y_val)

################## Predict #########################
output(X_test,idx_test,gbm,labelencoder)

