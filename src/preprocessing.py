from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocessing(X_train, X_test, y_train, tr_pipeline, select_feat, cache_file):
    # Engineer features
    X_train, X_test, _, _ = tr_pipeline(X_train,X_test, y_train, cache_file)

    # Create a validation set
    x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Select features
    select_feat.fit(X_train)
    
   
    x_trn = select_feat.transform(x_trn)
    x_val = select_feat.transform(x_val)
    X_test = select_feat.transform(X_test)

    # Encode y labels to integers
    le = LabelEncoder()
    le.fit(y_train)
    y_trn = le.transform(y_trn)
    y_val = le.transform(y_val)
    
    return x_trn, x_val, y_trn, y_val, X_test, le