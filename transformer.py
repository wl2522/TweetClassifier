import re
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

#Load the training and development datasets and separate the data from the labels

def data_load(training, test):

#Load the training and development datasets into lists

    with open(training, 'r', encoding='utf-8') as train_file:
        train_set = train_file.readlines()

    with open(test, 'r', encoding='utf-8') as test_file:
        test_set = test_file.readlines()
        
#Separate the labels from the tweets and store them as lists
    
    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()
    
    for tweet in train_set:
        data = tweet.split('\t')
        X_train.append(data[0])
        y_train.append(data[1][:-1])
        
    for tweet in test_set:
        data = tweet.split('\t')
        X_test.append(data[0])
        y_test.append(data[1][:-1])
        
    return (np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))

#Remove all the text that appears in shortened URLs after "http"


def remove_urls(X):
    
    X = re.sub(r': //t . co/[a-zA-Z0-9]{5,}', '', X)
    X = re.sub(r': //bit . ly/[a-zA-Z0-9]{5,}', '', X)
    X = re.sub(r': //tinyurl . com/[a-zA-Z0-9]{5,}', '', X)
    
    return X

#Detect if a tweet contains 2 or more consecutive question/exclamation marks, allowing for trailing spaces
#(returns 1 if found and 0 otherwise)


def repeated_punct(tweet):
    result = re.search(r'[(!\s)(?\s)]{4,}', tweet)
    
    if result is None:
        return 0
    else:
        return 1
    
#Detect if a tweet contains 3 or more consecutive instances of the same letter
#(returns 1 if found and 0 otherwise)
    
    
def repeated_letters(tweet):
    result = re.search(r'([a-zA-Z])\1{2,}', tweet)
    
    if result is None:
        return 0
    else:
        return 1
    
    
#Detect if a tweet is a retweet
#(returns 1 if the first three characters of the tweet are "RT " and 0 otherwise)


def retweet(tweet):
    if tweet[:3] == 'RT ':
        return 1
    else:
        return 0
    
    
#Detect if at least one other Twitter user was mentioned in the tweet
#(assume all Twitter usernames conform to the current 4 alphanumeric/underscore character minimum length)


def mentions(tweet):
    result = re.findall(r'@[a-zA-Z0-9_]{4,}', tweet)
    
    if len(result) == 0:
        return 0
    elif len(result) > 0:
        return 1

    
#Detect if any hashtags were included in the tweet
#(only count hashtags that conform to the rule of only containing alphanumeric characters and starting with a letter)


def hashtags(tweet):
    result = re.findall(r'#[a-zA-Z][a-zA-Z0-9_]{0,}', tweet)

    if len(result) == 0:
        return 0
    elif len(result) > 0:
        return 1


#Return the character length of the tweet

def tweet_length(tweet):
    
    return len(tweet)


#Detect if any URLs were included in the tweet
#(assume all URLs begin with "http")


def urls(tweet):
    result = re.findall(r'http', tweet)
    if len(result) == 0:
        return 0
    elif len(result) > 0:
        return 1    
    

#Detect if emojis were used in the tweet
#(assume that the associated unicode for all emojis in the dataset fall under these search pattern ranges)


def emoji_usage(tweet):
    
    emoji_pattern = re.compile(u'['
    u'\U0001F300-\U0001F5FF'
    u'\U0001F600-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+')
    result = re.findall(emoji_pattern, tweet)
    if len(result) == 0:
        return 0
    elif len(result) > 0:
        return 1


#Calculate the ratio of capital letters to the total number of letters in the tweet
#(not used in the final model)


def capital_ratio(tweet):
    capital_count = len(re.findall(r'[A-Z]', tweet))
    letter_count = len(re.findall(r'[a-zA-Z]', tweet))
    
    return capital_count/letter_count
    

#Tokenize the tweet to get a list of words used in the tweet with stop words excluded
#Calculate the average word length in the tweet


def avg_word_length(tweet):
    tokenizer = CountVectorizer(encoding='utf-8', stop_words='english', decode_error='ignore',
                             strip_accents='ascii').build_tokenizer()
    tokens = tokenizer(tweet)

    return len(''.join(tokens))/len(tokens)
    
#Apply each of the feature extraction functions to every sample
#Reshape each resulting array to form column vectors
#Concatenate the column vectors to form an extracted feature array


class FeatureExtractor(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        feature_functions = [repeated_punct, repeated_letters, retweet, mentions, hashtags,
                             tweet_length, urls, emoji_usage, avg_word_length]

        feature_values = list()
    
        for function in feature_functions:
            feature_values.append(np.vectorize(function)(X).reshape(-1, 1))

        return np.hstack(feature_values)
