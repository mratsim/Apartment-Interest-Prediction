from src.metrics import mlogloss
import numpy as np
import lightgbm as lgb
import time


#Ideally cross val split should be done before feature engineering, and feature engineering + selection should be done separately for each splits so it better mimics out-of-sample predictions
def cross_val_lgb(params, X, y, folds, metric):
    n =1
    num_rounds = 3000
    
    list_rounds = []
    list_scores = []

    
    for train_idx, valid_idx in folds:
        print('#################################')
        print('#########  Validating for fold:', n)

        lgb_train = lgb.Dataset(X[train_idx], label=y[train_idx])
        lgb_test = lgb.Dataset(X[valid_idx], label=y[valid_idx], reference=lgb_train)

        model = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_rounds,
                    valid_sets=lgb_test,
                    early_stopping_rounds=300,
                    verbose_eval=False)
        
        rounds = model.best_iteration
        score = model.best_score
        
        # Argh best_score doesn't exist yet in LightGBM: ticket open - https://github.com/Microsoft/LightGBM/issues/329
        #y_pred = model.predict(X[valid_idx], num_iteration=rounds)
        #score = mlogloss(y[valid_idx], y_pred)
        
        print('\nFold', n,'- best round:', rounds)
        print('Fold', n,'- best score:', score)
        
        list_rounds.append(rounds)
        list_scores.append(score)
        n +=1
    
    mean_score = np.mean(list_scores)
    std_score = np.std(list_scores)
    mean_round = np.mean(list_rounds)
    std_round = np.std(list_rounds)
    print('End cross validating',n-1,'folds') #otherwise it displays 6 folds
    print("Cross Validation Scores are: ", np.round(list_scores,3))
    print("Mean CrossVal score is: ", round(mean_score,3))
    print("Std Dev CrossVal score is: ", round(std_score,3))
    print("Cross Validation early stopping rounds are: ", np.round(list_rounds,3))
    print("Mean early stopping round is: ", round(mean_round,3))
    print("Std Dev early stopping round is: ", round(std_round,3))
    
    with open('./out/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'-lgb-cv.txt', 'a') as out_cv:
        out_cv.write("Cross Validation Scores are: " + str(np.round(list_scores,3)) + "\n")
        out_cv.write("Mean CrossVal score is: " + str(round(mean_score,3)) + "\n")
        out_cv.write("Std Dev CrossVal score is: " + str(round(std_score,3)) + "\n")
        out_cv.write("Cross Validation early stopping rounds are: " + str(np.round(list_rounds,3)) + "\n")
        out_cv.write("Mean early stopping round is: " + str(round(mean_round,3)) + "\n")
        out_cv.write("Std Dev early stopping round is: " + str(round(std_round,3)) + "\n")
    
    return mean_round
    