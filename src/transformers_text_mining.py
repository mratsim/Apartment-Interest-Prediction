import re

def tr_desc_mining(train, test, y, folds, cache_file):
    def number_caps(x):
        return sum(1 for c in x if c.isupper())/float(len(x)+1)
    def _trans(df):
        df['redacted'] = 0
        df['redacted'].ix[df['description'].str.contains('website_redacted')] = 1
        
        df['email'] = 0
        df['email'].ix[df['description'].str.contains('@')] = 1
        
        # Phone RegExp
        reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
        def try_and_find_nr(description):
            if reg.match(description) is None:
                return 0
            return 1
        
        return df.assign(
            number_caps = df['description'].apply(number_caps),
            number_lines = df['description'].apply(lambda x: x.count('<br /><br />')),
            phone_nr = df['description'].apply(try_and_find_nr)
        )
    return _trans(train),_trans(test), y, folds, cache_file