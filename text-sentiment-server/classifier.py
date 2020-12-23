# run this cell to import nltk
import nltk
import sys
import os
import logging
import pickle
import numpy as np
from nltk.corpus import twitter_samples 
from utils import process_tweet, build_freqs, extract_features, sigmoid, gradientDescent

'''
Init and configuration
'''
logger = logging.getLogger('text-assessment-server')
MODEL_FOLDER = 'model'
THETA_FILE = MODEL_FOLDER + '/theta.bin'
FREQS_FILE = MODEL_FOLDER + '/freqs.bin'

# Provides working data set
def get_dataset():
    # download Twitter sample dataset
    logger.info("Retrieving dataset.")
    nltk.download('twitter_samples')

    # download stopwords
    nltk.download('stopwords')

    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    return all_positive_tweets, all_negative_tweets

# Splits data set into training / test data
def split_dataset(pos_data, neg_data):
    # split the data into two pieces, one for training and one for testing (validation set)
    split_factor = 0.8
    logger.info("Splitting dataset, split factor: " + str(split_factor))
    p_len = int(len(pos_data) * split_factor)
    n_len = int(len(neg_data) * split_factor)

    test_pos = pos_data[p_len:]
    train_pos = pos_data[:p_len]

    test_neg = neg_data[n_len:]
    train_neg = neg_data[:n_len]

    # combine positive and negative values
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

     # combine positive and negative labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)  

    return train_x, test_x, train_y, test_y

# Trains logistic regression classifier
def train_model():
    # retrieve and split data set
    pos_data, neg_data = get_dataset()
    train_x, test_x, train_y, test_y = split_dataset(pos_data, neg_data)

    # create frequency dictionary
    freqs = build_freqs(train_x, train_y)
    logger.info("Completed frequency matrix, total length: " + str(len(freqs.keys())))

    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))

    for i in range(len(train_x)):
        X[i, :]= extract_features(train_x[i], freqs)

    # training labels corresponding to X|
    Y = train_y

    # Apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
    logger.info(f"The cost after training is {J:.8f}.")
    logger.info(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

    # calculate accuracy
    accuracy = test_model(test_x, test_y, freqs, theta)
    logger.info("Accuracy on test dataset: " + str(accuracy))

    return freqs, theta

# Loads model from binary files
def load_model():
    freqs = None
    theta = None

    if(os.path.exists(FREQS_FILE)):
        with open(FREQS_FILE, 'rb') as fp:
            freqs = pickle.load(fp)
    
    if(os.path.exists(THETA_FILE)):
        with open(THETA_FILE, 'rb') as fp:
            theta = pickle.load(fp)

    return freqs, theta

# Saves model in binary files
def save_model(freqs, theta):
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    with open(FREQS_FILE, 'wb') as fp:
        pickle.dump(freqs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(THETA_FILE, 'wb') as fp:
        pickle.dump(theta, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Tests trained classifier
def test_model(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_phrase(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = np.sum((np.asarray(y_hat) == np.squeeze(test_y))) / len(y_hat)
    
    return accuracy

# Predicts class of text
def predict_phrase(phrase, freqs, theta):
    '''
    Input: 
        phrase: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a text being positive or negative
    '''
    # extract the features of the text and store it into x
    x = extract_features(phrase, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred

def init():
    # try to load model
    freqs, theta = load_model()

    # if model is not available yet, let's re-train that
    if (freqs is None) or (theta is None):
        # training model from scratch
        logger.info("No pre-trained model available, training now...")
        freqs, theta = train_model()

         # save model in binary files
        save_model(freqs, theta)
    
    logger.info("Loaded model.")
    return freqs, theta


