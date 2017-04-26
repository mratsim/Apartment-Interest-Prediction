import pandas as pd

def tr_remove_outliers(train, test, y, folds, cache_file):

    test["bathrooms"].loc[19671] = 1.5
    test["bathrooms"].loc[22977] = 2.0
    test["bathrooms"].loc[63719] = 2.0
    test["bathrooms"].loc[17808] = 2.0
    test["bathrooms"].loc[22737] = 2.0
    test["bathrooms"].loc[837] = 2.0
    test["bedrooms"].loc[100211] = 5.0
    test["bedrooms"].loc[15504] = 4.0
    train["price"] = train["price"].clip(upper=13000)
    
    return train, test, y, folds, cache_file