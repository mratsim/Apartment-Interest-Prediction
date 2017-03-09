import operator as op

def tr_tfidf_features(train, test, y, cache_file):
    def _trans(df):
        return df.assign(
            joined_features = df['features'].str.join(', '),
            joined_feat_underscore = df['features'].apply(lambda x:list(map(op.methodcaller("replace", ' ', '_'),x))).str.join(', ')
        )
    return _trans(train), _trans(test), y, cache_file