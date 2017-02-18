import pandas as pd

####### Debug Transformer ###########
# Use this transformer anywhere in your Pipeline to dump your dataframe to CSV
def tr_dumpcsv(train,test, y):
    train.to_csv('./dump_train.csv')
    test.to_csv('./dump_test.csv')
    return train,test, y