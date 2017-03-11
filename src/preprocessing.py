from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.star_command import feat_selection

def preprocessing(X_train, X_test, y_train, tr_pipeline, select_feat, folds, cache_file):
    # Engineer features
    X_train, X_test, _, _, _ = tr_pipeline(X_train,X_test, y_train, folds, cache_file)
    
    # Create a validation set
    x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Select features - validation
    x_trn, x_val = feat_selection(select_feat, x_trn, x_val, y_trn)
    
    # Select features - prediction
    X_train, X_test = feat_selection(select_feat, X_train, X_test, y_train)

    print('X_train has', X_train.shape[1], 'features')
    
    # Encode y labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_train)
    y_trn = le.transform(y_trn)
    y_val = le.transform(y_val)
    
    return x_trn, x_val, y_trn, y_val, le, X_train, X_test, y