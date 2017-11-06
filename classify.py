
# coding: utf-8

# In[6]:

import sys
import re

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from transformer import *


# In[13]:

#Train and save the model that provided the best training accuracy

def best_model(train, test):
    X_train, X_test, y_train, y_test = data_load(train, test)
    
    n_vect = CountVectorizer(encoding='utf-8', stop_words='english', ngram_range=(1, 2),
                             decode_error='ignore', strip_accents='ascii')
    
#Apply the remove_urls function to each tweet in the dataset
    
    X_train = np.vectorize(remove_urls)(X_train)
    X_test = np.vectorize(remove_urls)(X_test)
    
#Combine the array of vectorized text with an array containing additional writing style/content features
#Fit the model to the training data and save the model as "model.pkl"
    
    
    union = FeatureUnion([('n_vect', n_vect), ('style_features', FeatureExtractor())])
    nb = MultinomialNB(alpha=.25)
    X_train_vect = union.fit_transform(X_train)
    nb.fit(X_train_vect, y_train)

    joblib.dump(nb, 'model.pkl')

#Vectorize the training/test data and save it as "test.pkl" along with the vectorizer
#Return the test accuracy of the model
    
    n_vect.fit(X_train)
    X_test_vect = union.transform(X_test)
    joblib.dump((n_vect.transform(X_train), y_train, X_test_vect, y_test, n_vect), 'test.pkl')
    
    return (nb.score(X_test_vect, y_test))


# In[3]:

try:
    print('Best model test accuracy: ', best_model(sys.argv[1], sys.argv[2]))

except:
    pass

