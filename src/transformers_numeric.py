# This transformer extracts the number of photos
def tr_numphot(train, test, y):
    def _trans(df):
        return df.assign(NumPhotos = df['photos'].str.len())
    return _trans(train), _trans(test), y
    
# This transformer extracts the number of features
def tr_numfeat(train, test, y):
    def _trans(df):
        return df.assign(NumFeat = df['features'].str.len())
    return _trans(train), _trans(test), y
    
# This transformer extracts the number of words in the description
def tr_numdescwords(train, test, y):
    def _trans(df):
        return df.assign(
            NumDescWords = df["description"].apply(lambda x: len(x.split(" ")))
            )
    return _trans(train), _trans(test), y