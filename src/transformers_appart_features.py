def tr_tfidf_features(train, test, y, cache_file):
    def _trans(df):
        return df.assign(
            joined_features = df['features'].str.join(', ')
        )
    return _trans(train), _trans(test), y, cache_file