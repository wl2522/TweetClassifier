Wilson Lui

wl2522@columbia.edu

Homework 1

This project contains four files:

1) transformer.py contains auxiliary functions for loading the training and test datasets in the correct format and extracting features from the data.

2) classify.py contains a function for training a Naive Bayes classifier and scoring its accuracy on a test set. This function can be run from the command line using the command "python3 classify.py <training_data> <test_data>", where both the training and test data are expected to be text files where each line contains a single tweet followed by its class label with a tab separating the two. When this command is run, the trained model is saved as a pickle file named "model.pkl". The test data is saved in a vectorized form as a pickle file named "test.pkl". Lastly, the accuracy achieved on the test dataset is displayed.

3) analyze.py contains a function that displays the top 20 features and their associated F-values as well as a contingency matrix that shows how well the classifier performed on the test dataset. This function can be run from the command line using the command "python3 analyze.py <model> <test_data>", where the model and vectorized test data are expected to be pickle files. For example, if this command is run after classify.py saves its output, then the command would be "python3 analyze.py model.pkl test.pkl".

4) hw1.py is a wrapper file that contains functions for finding the best classifier and optimal parameters for unigram, bigram, and trigram models. This is done through a grid search over different combinations of classifiers and parameter settings. For each optimal n-gram model, the training accuracy, the top 20 features found, and a contingency matrix that shows how well the model performed on the training dataset are displayed. Lastly, it uses the functions contained in classify.py and analyze.py to train the best performing model and print its test accuracy, the top 20 features, and the contingency matrix obtained from its test dataset predictions.

My classifier uses unigram and bigram features with English stop words and all punctuation besides question and exclamation marks removed. In addition, it makes use of several writing content and style features. For example, it can detect the presence of repeated exclamation and question marks, emojis, URLs, retweets, mentions, and hashtags. It also finds the length of each tweet and its average word length.

Some limitations of the model include the fact that a Twitter user's political affiliation can only be predicted based off one tweet. This is especially limiting if that one tweet does not pertain to politics. Similarly, the model does not know anything else about the user outside of that one tweet. In addition, the model doesn't know what's contained in the URLs that are found in the tweets because the domain names are hidden by Twitter's URL shortener. Therefore, the model simple replaces each URL with "http" to denote the presence of a URL. This avoids the problem of having otherwise meaningless URL text being treated as features in the dataset.
