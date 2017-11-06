
# coding: utf-8

# In[12]:

import sys

import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# In[17]:

#Return the 20 most important features based on the training dataset
#Return the confusion matrix produced when the model makes predictions on the test dataset

def contingency_matrix(model, vect_test):
    
#Load the model, the vectorized training and test data, and the vectorizer used to create them

    best_model = joblib.load(model)
    X_train, y_train, X_test, y_test, n_vect = joblib.load(vect_test)

#Find the 20 largest F-values and their associated feature names and column indices

    n_features = SelectKBest(score_func=f_classif, k=20)
    n_features.fit(X_train, y_train)
    top_20 = np.argpartition(n_features.scores_, -20)[-20:]
    features = list(zip(n_features.scores_[top_20],
                        np.array(n_vect.get_feature_names())[top_20]))
    features.sort(reverse=True)
    
    y_pred = best_model.predict(X_test)
    
    return (features, confusion_matrix(y_test, y_pred))
    


# In[31]:

try:
    top_features, conf_matrix = contingency_matrix(sys.argv[1], sys.argv[2])

    print('Top 20 n-gram features: \n')

    for feature in top_features:
        print(feature)

    print('Best n-gram model confusion matrix: \n', conf_matrix)
    
except:
    pass

