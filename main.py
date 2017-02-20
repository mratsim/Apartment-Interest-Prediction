# coding: utf-8

# Captain obvious
import numpy as np
import pandas as pd

# feature preprocessing
from sklearn.preprocessing import RobustScaler,StandardScaler, OneHotEncoder

# Custom helper functions
from src.star_command import feat_extraction_pipe
from main_output import output
from main_train import train_lgb
from src.preprocessing import preprocessing

# Mesure time
from timeit import default_timer as timer

# Set random seed for reproducibility
np.random.seed(1337)

# Start timer
start_time = timer()

# # Import data
df_train = pd.read_json(open("./data/train.json", "r"))
df_test = pd.read_json(open("./data/test.json", "r"))

print('Input training data has shape: ',df_train.shape)
print('Input test data has shape:     ',df_test.shape)

X = df_train
X_test = df_test
y = df_train['interest_level']
idx_test = df_test['listing_id']

###### TODO ###########
# Bucket nombre de chambres et de bathrooms
# Retirer numéro de rue - DONE ?
# Imputer les rues sans géoloc
# Quartier (centre le plus proche)
# Distance par rapport au centre
# Clusteriser la latitude/longitude
# manager skill (2*high + medium)
# TFIDF - Naive Bayes

#######################

# # Command Center

from src.transformers_numeric import tr_numphot, tr_numfeat, tr_numdescwords
from src.transformers_time import tr_datetime
from src.transformers_debug import tr_dumpcsv
from src.transformers_nlp_tfidf import tr_tfidf_lsa_lgb
from src.transformers_categorical import tr_enc_dispAddr, tr_enc_manager, tr_enc_building, tr_enc_streetAddr

# Feature engineering - sequence of transformations
tr_pipeline = feat_extraction_pipe(
    tr_numphot,
    tr_numfeat,
    tr_numdescwords,
    tr_datetime,
    tr_enc_dispAddr,
    tr_enc_manager,
    tr_enc_building,
    tr_enc_streetAddr,
    tr_tfidf_lsa_lgb
    #tr_dumpcsv
)

# Feature selection - features to keep
select_feat = [
    (["bathrooms"],None),
    (["bedrooms"],None),
    (["latitude"],None),
    ("longitude",None),
    (["price"],None),
    ("NumDescWords",None),
    ("NumFeat",None),
#    ("Created_Year",None), #Every listing is 2016
    ("Created_Month",None),
    ("Created_Day",None),
    ("Created_Hour",None),
    ("Created_DayOfWeek",None),
    #("tfidf_high",None),
    #("tfidf_medium",None),
    #("tfidf_low",None),
    (["encoded_display_address"],OneHotEncoder(sparse='False',handle_unknown='ignore')), #Categorical feature
    (["encoded_manager_id"],OneHotEncoder(sparse='False',handle_unknown='ignore')), #Categorical feature
    (["encoded_building_id"],OneHotEncoder(sparse='False',handle_unknown='ignore')), #Categorical feature
    (["encoded_street_address"],OneHotEncoder(sparse='False',handle_unknown='ignore')) #Categorical feature
]

# Currently LightGBM core dumps on categorical data, deactivate in the transformer

################ Preprocessing #####################
cache_file = './cache.db'

x_trn, x_val, y_trn, y_val, X_test, labelencoder = preprocessing(
   X, X_test, y, tr_pipeline, select_feat, cache_file)

############ Train and Validate ####################
print("############ Final Classifier ######################")
gbm, metric = train_lgb(x_trn, x_val, y_trn, y_val)

################## Predict #########################
output(X_test,idx_test,gbm,labelencoder, metric)

end_time = timer()
print("################## Success #########################")
print("Elapsed time: %s" % (end_time - start_time))
