# Logic to load data from cache file
# Idea: use a cache parameter for every transformer. Change pipe map to starmap, remove y and cache from "preprocessing fonction"

import os
import shelve
from pickle import HIGHEST_PROTOCOL
 
def load_from_cache(cache_file, key_train, key_test):
    if os.path.isfile(cache_file):
        with shelve.open(cache_file, flag='r', protocol=HIGHEST_PROTOCOL) as db:
            if (key_train in db) and (key_test in db):
                print('#### Using cached data ####')
                dict_train = db[key_train]
                dict_test = db[key_test]
                db.close()
                return dict_train, dict_test
    return None, None

def save_to_cache(cache_file, key_train, key_test, dict_train, dict_test):
    with shelve.open(cache_file, flag='c', protocol=HIGHEST_PROTOCOL) as db:
        db[key_train] = dict_train
        db[key_test] = dict_test
        db.close()