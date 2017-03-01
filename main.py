# coding: utf-8

# Captain obvious
import numpy as np
import pandas as pd

# feature preprocessing
from sklearn.preprocessing import Normalizer, OneHotEncoder, Imputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import category_encoders as ce

#Dimensionality reduction
from sklearn.decomposition import TruncatedSVD, NMF

# feature selection
# from sklearn.feature_selection import SelectFromModel, RFECV
# from sklearn.model_selection import StratifiedKFold

# Custom helper functions
from src.star_command import feat_extraction_pipe
from main_output import output
from main_train import training_step
from src.preprocessing import preprocessing
from src.metrics import mlogloss

# Mesure time
from timeit import default_timer as timer
import time

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
# Bucket nombre de chambres et de bathrooms - DONE display addresse
# Retirer numéro de rue - DONE ?
# Imputer les rues sans géoloc
# Quartier (centre le plus proche)
# Distance centre NY
# Distance par rapport au centre
# Clusteriser la latitude/longitude
# manager skill (2*high + medium)
# TFIDF - Naive Bayes and/or SVM
# Tout passer en minuscule
# Remplacer les ave par avenue, n par north etc
# Redacted dans la website description (sauf si beautiful soup le fait)
# Building interest
# lignes de métro: A,C,E,L,1,2,3 trains
# Metro lines from geocoding ?
# Rounding floats to reduce noise
# Extract toilets
# price per room
#  On the Rent Hop website you can notice that when browsing listings only the first two images are displayed. Only when you click on the listing do you see the rest of the images.
# Size of image (x * x and in MB)
# Exif
# Imputation, depuis building id, street address -> display address, latitude, longitude
# Public Holiday
# Season
# Business quarter
# Business hours, lunch hours, early morning, evening, late evening
# Lag: high interest in the past hour/day
# Lag: low interest in the past hour/day
# Semaine juste avant/après les congés

#######################

# # Command Center

from src.transformers_numeric import tr_numphot, tr_numfeat, tr_numdescwords, tr_log_price, tr_bucket_rooms, tr_price_per_room, tr_split_bath_toilets
from src.transformers_time import tr_datetime
from src.transformers_debug import tr_dumpcsv
from src.transformers_nlp_tfidf import tr_tfidf_lsa_lgb
from src.lib_sklearn_transformer_nlp import NLTKPreprocessor, HTMLPreprocessor
from src.transformers_appart_features import tr_tfidf_features
from src.transformers_categorical import tr_managerskill, tr_bin_buildings_mngr
from src.transformers_categorical import tr_encoded_manager, tr_encoded_building, tr_encoded_disp_addr, tr_encoded_street_addr
from src.transformers_geoloc import tr_clustering

# Feature extraction - sequence of transformations
tr_pipeline = feat_extraction_pipe(
    tr_numphot,
    tr_numfeat,
    tr_numdescwords,
    tr_datetime,
    tr_split_bath_toilets,
    tr_tfidf_lsa_lgb,
    tr_managerskill,
    #tr_log_price,
    tr_bucket_rooms,
    tr_bin_buildings_mngr,
    tr_price_per_room,
    tr_tfidf_features,
    tr_encoded_manager,
    tr_encoded_building,
    tr_encoded_disp_addr,
    tr_encoded_street_addr,
    #tr_clustering,
    tr_price_per_room
    #tr_dumpcsv
)

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

#can't bag plurals with singular
vocab_metro = {
    'metro':0,
    'metros':1,
    'train':2,
    'trains':3,
    'line':4,
    'lines':5, 
}
vocab_metro_lines = {
    '1':0,
    '2':1,
    '3':2,
    '4':3,
    '5':4,
    '6':5,
    '7':6,
    'A':7,
    'C':8,
    'E':9,
    'B':10,
    'D':11,
    'F':12,
    'M':13,
    'G':14,
    'J':15,
    'Z':16,
    'L':17,
    'S':18,
    'N':19,
    'Q':20,
    'R':21,
    'W':22
}

# Feature selection - features to keep
select_feat = [
    ("bathrooms",None),
    #('bathrooms_only',None),
    ('toilets_only',None),
    ("bucket_bath",None), #
    (["bedrooms"],None),
    ('bucket_bed',None),  # 
    #('rooms_sum',None),
    #('rooms_diff',None),
    #('price_per_totalrooms',None), #
    #('rooms_ratio', None), #
    (["latitude"],None),
    (["longitude"],None),
    #("latitude_cluster",None),
    #("longitude_cluster",None),
    #(['cluster'],None), #
    #('log_price',None),
    (["price"],None),
    ('price_per_room',None),
    (["NumDescWords"],None),
    (["NumFeat"],None),
    (["NumPhotos"],None),
    #("Created_Year",None), #Every listing is 2016
    (["Created_Month"],None),
    (["Created_Day"],None),
    (["Created_Hour"],None),
    ('listing_id',None),
    #(["Created_DayOfWeek"],None), #
    #('Created_DayOfYear',None),
    #('Created_WeekOfYear',None), #
    #('Created_D_cos',None),
    #('Created_D_sin',None),
    ('Created_DoW_cos',None),
    ('Created_DoW_sin',None),
    ('Created_DoY_cos',None),
    ('Created_DoY_sin',None),
    ('Created_WoY_cos',None),
    ('Created_WoY_sin',None),
    ('Created_Weekend',None),
    #('Time_passed',None), #
    ('Is_Holiday',None),
    ('Is_SchoolHoliday',None),
    ("tfidf_high",None),
    ("tfidf_medium",None),
    ("tfidf_low",None),
    ("encoded_display_address",None),
    ("display_address",CountVectorizer()), ##
    #("encoded_street_address",None),
    #("street_address",CountVectorizer()),
    (["encoded_manager_id"],None),
    ("manager_id",CountVectorizer()),
    (["encoded_building_id"],None),
    ("building_id",CountVectorizer()),
    ('mngr_percent_high',None),
    ('mngr_percent_low',None),
    ('mngr_percent_medium',None),
    ('mngr_skill',None),
    ('Bin_Buildings',None),
    ('Bin_Managers',None),
    ("joined_features", CountVectorizer( ngram_range=(1, 2), #1,2 ?
                                       stop_words='english',
                                       max_features=200)),
    ("description", [TfidfVectorizer(max_features=2**16,
                             min_df=2, stop_words='english',
                             use_idf=True),
                    TruncatedSVD(2), # 2 or 3
                    Normalizer(copy=False)]),
    #("CleanDesc",[HTMLPreprocessor(),NLTKPreprocessor(),
    #                TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)]
    #)
    ("description", CountVectorizer(vocabulary=vocab_metro,binary=True)),
    #("description", CountVectorizer(vocabulary=vocab_metro_lines,binary=True, lowercase=False))
]
                    


# Currently LightGBM core dumps on categorical data, deactivate in the transformer

################ Preprocessing #####################
cache_file = './cache.db'

# Sorry for the spaghetti code
x_trn, x_val, y_trn, y_val, labelencoder, X_train, X_test, y_train = preprocessing(
   X, X_test, y, tr_pipeline, select_feat, cache_file)

############ Train and Validate ####################
print("############ Final Classifier ######################")
clf, metric, n_stop = training_step(x_trn, x_val, y_trn, y_val, X_train, y_train)

################## Predict #########################
output(X_test,idx_test,clf,labelencoder, n_stop, metric)

with open('./out/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'-features.txt', "w") as text_file:
    for item in select_feat:
        text_file.write("{}\n".format(item))

end_time = timer()
print("################## Success #########################")
print("Elapsed time: %s" % (end_time - start_time))
