import pandas as pd
import time

####### Predict and format output #######
def output(X_test, listing_id, classifier, LabelEncoder, metric):
    print('Start predicting...')
    
    # print('Type of data - Test - check especially for categorical')
    # print(X_test.dtypes)
    
    predictions = classifier.predict(X_test, num_iteration=classifier.best_iteration)
    # predictions = classifier.predict(X_test, ntree_limit=classifier.best_ntree_limit)

    
    #debug
    print('\n\nPredictions done. Here is a snippet')
    print(LabelEncoder.classes_)
    print(predictions)
    
    result = pd.DataFrame({
        'listing_id': listing_id,
        LabelEncoder.classes_[0]: [row[0] for row in predictions], 
        LabelEncoder.classes_[1]: [row[1] for row in predictions],
        LabelEncoder.classes_[2]: [row[2] for row in predictions]
        })
    result.to_csv('./out/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'.csv', index=False)