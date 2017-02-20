from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.command_center import feat_selection

def preprocessing(X_train, X_test, y_train, tr_pipeline, select_feat, cache_file):
    # Engineer features
    X_train, X_test, _, _ = tr_pipeline(X_train,X_test, y_train, cache_file)

    # Create a validation set
    x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Select features 
    x_trn, x_val, x_test = feat_selection(select_feat, x_trn, x_val, X_test)
    
    # Encode y labels to integers
    le = LabelEncoder()
    le.fit(y_train)
    y_trn = le.transform(y_trn)
    y_val = le.transform(y_val)
    
    return x_trn, x_val, y_trn, y_val, x_test, le