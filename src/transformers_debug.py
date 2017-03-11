####### Debug Transformer ###########
# Use this transformer anywhere in your Pipeline to dump your dataframe to CSV
def tr_dumpcsv(train,test, y, folds, cache_file):
    train.to_csv('./dump_train.csv')
    test.to_csv('./dump_test.csv')
    return train,test, y