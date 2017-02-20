# inspired by: https://joshbohde.com/blog/functional-python
# Transformations do not take extra arguments so no need for partial or starmap

from functools import reduce
from itertools import starmap
# from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


############ Helpers Functional Programming####################
#compose list of functions (chained composition)
def compose(*funcs):
    def _compose(f, g):
        # functions are expecting X,y not (X,y) so must unpack with *g
        return lambda *args, **kwargs: f(*g(*args, **kwargs))
    return reduce(_compose, funcs)

# zipWith
def zip_with(f, list_of_tuple):
    return starmap(f, zip(*list_of_tuple))

# Parallel zipWith
# def par_zip_with(f, list_of_tuple, pool):
#     return pool.starmap(f, zip(*list_of_tuple))

############# Helper pandas ###################
def vconcat(*list_of_df):
    return pd.concat([*list_of_df], axis=1)


# pipe functions, reverse the order so that it's in the usual FIFO function order
def feat_extraction_pipe(*funcs):
    return compose(*reversed(funcs))

# WARNING: Beware of modifying the original columns of dataframe in place in a  multiprocessing context.
# It can create race condition, and unexpected behaviour

def feat_selection(ft_selection,df_train,df_val,df_test):
    trn = df_train.copy()
    val = df_val.copy()
    tst = df_test.copy()
    
    ft_selection = [(trn,val,tst,label,transfo) for (label,transfo) in ft_selection]
    
    tuples_trn_val_test = starmap(feat_transformation,ft_selection)
    trn, val, tst = zip_with(vconcat,tuples_trn_val_test)
    
    ## Multiprocessing is slower and obfuscate error (MemoryError or TrhadLock instead of the real issue)
    ## selecting 18 features : 44-47 seconds for no MP vs 2x63 with MP. GIL/pickle issue ?
    # with Pool(cpu_count()) as mp:
    #     tuples_trn_val_test = mp.starmap(feat_transformation,ft_selection)
    #     trn, val, tst = par_zip_with(vconcat,tuples_trn_val_test, mp)
    return trn, val, tst

def feat_transformation(train,valid, test,sCol,Transformer=None):
    if Transformer is None:
        return (train[sCol], valid[sCol], test[sCol])

    def _trans(df, sCol, Transformer,flag):
        if flag == "fit_transform":
            transformed = Transformer.fit_transform(df[sCol]).T
        elif flag == "transform":
            transformed = Transformer.transform(df[sCol]).T
        elif flag == "fit":
            transformed = Transformer.fit(df[sCol]).T
        #else Raise exception
        
        if isinstance(sCol, list):
            label = sCol[0]
        else:
            label = sCol
        
        # To keep the index, we need to assign to the original dataframe and then extract again
        # OneHotEncoder and LabelBinarizer do not have the same interface ....
        feature_list = []
        n = 0
        feat_label = ''
        for serie in transformed:
            if isinstance(Transformer, OneHotEncoder):
                feat_label = label + '_' + Transformer.active_features_[n]
            else:
                try:
                    feat_label = label + '_' + Transformer.classes_[n]
                except:
                    feat_label = label
            df[feat_label] = serie
            feature_list.append(feat_label)
        return df[feature_list]
    
    trn = train.copy()
    val = valid.copy()
    tst = test.copy()
    
    trn = _trans(trn, sCol, Transformer,'fit_transform')
    val = _trans(val, sCol, Transformer,'transform')
    tst = _trans(tst, sCol, Transformer,'transform')
    # If you have unseen data issue when transforming test, preprocess your data, don't leak. (Label don't exist in train
    
    #We only send back what we need
    return (trn, val, tst)
