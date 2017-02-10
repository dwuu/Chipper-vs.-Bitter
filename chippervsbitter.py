#Daniel Wu
#Chipper Vs. Bitter

import pandas as pd
import numpy as np
import itertools
import string as s
import random

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt;
from sklearn.metrics import confusion_matrix

def load_data(fname):
  """
  Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
  You can access the label by calling df['label'], the content by df['content']
  the sentiment by df['sentiment']
  """
  return pd.read_csv(fname)

def initialize_stuff(df):
	"""
	Takes in the training data, extracts words from each tweet to create a dictionary that
	will act as a the feature vector (theta) for the classifier h(X,theta).  Returns the dictionary
	and the feature matrix for all tweets.
	"""

	dic = {}
	idx = 0

	#dictionary size is 1360
	fm = np.zeros((df['content'].size, 1360))


	tweet_number = 0
	#for every tweet in the training set...
	for tweet in df['content']:
		#replace punctuation with whitespace
	  	transtable = str.maketrans(".,?!\";:#$%^&*()-=+_/\|[]{}<>", "                            ")
	  	tweet = tweet.translate(transtable)

	  	#removes numbers and apostrophe's without replacing with whitespace
	  	transtable = str.maketrans("","", "'1234567890")
	  	tweet = tweet.translate(transtable).lower()

	 	#add words to dictionary
	  	for word in tweet.split():
	  		if word not in dic:
	  			if len(word)<2: #ignores single letters that likely have little meaning, like "a"
	  				continue
	  			if word[0] == "@": #ignores @mentions
	  				continue

	  			dic[word] = idx
	  			idx+=1

	  	#set the feature vectors' values to the number of times a word in the dictionary appears in the tweet
	  	for key, value in dic.items():
	  		if key in tweet.split():
	  			fm[tweet_number][value] = tweet.split().count(key)

	  	tweet_number += 1

	return fm,dic

def performance(y_true, y_pred, metric="accuracy"):
	"""
		Calculates the performance metric based on the agreement between the
		true labels and the predicted labels
		Input:
		  y_true- (n,) array containing known labels
		  y_pred- (n,) array containing predicted scores
		  metric- string option used to select the performance measure
		Returns: the performance as a np.float64
	"""
	if metric == "accuracy":
		return metrics.accuracy_score(y_true, y_pred)

	if metric == "f1-score":
		return metrics.f1_score(y_true, y_pred)

	if metric == "precision":
		return metrics.precision_score(y_true, y_pred)

	if metric == "sensitivity": 
		return metrics.recall_score(y_true, y_pred)

	if metric == "auroc":
		return metrics.roc_auc_score(y_true, y_pred)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
	"""
		Splits the data, X and y, into k-folds and runs k-fold crossvalidation:
		training a classifier on K-1 folds and testing on the remaining fold.
		Calculates the k-fold crossvalidation performance metric for classifier
		clf by averaging the performance across folds.
		Input:
		clf- an instance of SVC()
		X- (n,d) array of feature vectors, where n is the number of examples
		   and d is the number of features
		y- (n,) array of binary labels {1,-1}
		k- int specificyin the number of folds (default=5)
		metric- string specifying the performance metric (default='accuracy',
				 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
				 and 'specificity')
		Returns: average 'test' performance across the k folds as np.float64
	"""

	skf = StratifiedKFold(n_splits=k)
	skf.get_n_splits(X,y)

	total = 0

	#for each "fold"
	for train_indexes, test_indexes in skf.split(X, y):
		X_train = X[train_indexes]
		y_train = y[train_indexes]

		X_test = X[test_indexes]
		y_test = y[test_indexes]

		#train the classifier
		clf.fit(X_train, y_train)

		#classifications that the classifier calculates
		y_pred = clf.predict(X_test)

		#if auroc, set the y_pred parameter to be y_score instead
		if metric == "auroc": 
			y_score = clf.decision_function(X_test)
			y_pred = y_score

		#Sums the performance (based on the metric) for each split 
		total += performance(y_test, y_pred, metric)

	#return the average performance of using each split as the test set
	return total/k

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
	"""
	  Sweeps different settings for the hyperparameter of a linear-kernel SVM,
	  calculating the k-fold CV performance for each setting on X, y.
	  Input:
		X- (n,d) array of feature vectors, where n is the number of examples
		   and d is the number of features
		y- (n,) array of binary labels {1,-1}
		k- int specifying the number of folds (default=5)
		metric- string specifying the performance metric (default='accuracy',
				 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
				 and 'specificity')
		C_range - an array with all C values to be checked for
	  Returns the parameter value for linear-kernel SVM, that 'maximizes' the
	  average 5-fold CV performance.
	"""
	best_c = C_range[0]
	best_val = 0

	for c in C_range:
		#new instance of Support Vector Classifier
		clf = SVC(kernel='linear', C=c, class_weight='balanced')

		#calculates cross-validation performance of each classifier(that differ in C)
		perf = cv_performance(clf,X,y,k,metric)
		if perf > best_val:
			best_val = perf
			best_c = c

	#returns C that optimizes performance
	return best_c

def generate_classifiers(X,df):
	#X is a feature matrix for x.size tweets

	#labels vectors (given classifications for each tweet):
	#y_love considers love labels to be 1, and everything else to be -1
	#y_hate considers hate labels to be 1, and everything else to be -1
	#y_sadness considers sadsness labels to be 1, and everything else to be -1

	y_love = []
	y_hate = []
	y_sadness = []

	for i in range(0,300):
		if df['sentiment'][i] == 'love':
			y_love.append(1)
			y_hate.append(-1)
			y_sadness.append(-1)
		if df['sentiment'][i] == 'hate':
			y_love.append(-1)
			y_hate.append(1)
			y_sadness.append(-1)
		if df['sentiment'][i] == 'sadness':
			y_love.append(-1)
			y_hate.append(-1)
			y_sadness.append(1)

	y_love = np.asarray(y_love)
	y_hate = np.asarray(y_hate)
	y_sadness = np.asarray(y_sadness)

	#Provides 7 different magnitudes of potential values for hyperparameter C

	C_range = [10e-4, 10e-3, 10e-2, 10e1, 10e0, 10e1, 10e2]

	#determine the best C values for each classifier, optimized based on the AUROC metric,
	#or Area Under ROC (curve that compares true positives and false positives)

	c_love = select_param_linear(X, y_love, C_range = C_range, metric = "auroc")
	c_hate = select_param_linear(X, y_hate, C_range = C_range, metric = "auroc")
	c_sadness = select_param_linear(X, y_sadness, C_range = C_range, metric = "auroc")

	#Using the optimal C values,

	love = SVC(kernel='linear', C=c_love, probability = True, class_weight='balanced')
	love.fit(X, y_love)

	hate = SVC(kernel='linear', C=c_hate, probability = True, class_weight='balanced')
	hate.fit(X, y_hate)

	sadness =SVC(kernel='linear', C=c_sadness, probability = True, 	class_weight='balanced')
	sadness.fit(X, y_sadness)

	#returns the 3 one vs all classifiers, each trained using their respective adjusted labels
	return love, hate, sadness

#Looks at each classifier's confidence (using SVC.predict_proba) in classifying the test data 
def calculate_prediction(love, hate, sadness, test_fm):
	x = len(test_fm)
	l_prob = love.predict_proba(test_fm)
	h_prob = hate.predict_proba(test_fm)
	s_prob = sadness.predict_proba(test_fm)

	#generates len(test_fm) * 3 matrix, where for each tweet, the 3 columns have 0 (love) ,1 (hate),2 (sadness)
	#sorted by order of highest probability.
	order = []

	for i in range (0, x):
		temp = []
		temp.append(l_prob[i][0])
		temp.append(h_prob[i][0])
		temp.append(s_prob[i][0])
		temp = sorted(range(len(temp)), key=temp.__getitem__)
		order.append(temp)

	prediction = []

	#these are the predictions made by each classifier for every tweet
	l = love.predict(test_fm)
	h = hate.predict(test_fm)
	s = sadness.predict(test_fm)

	for i in range(0, x):
		temp = []
		temp.append(l[i])
		temp.append(h[i])
		temp.append(s[i])


		#Use the highest confidence classifier if that classifier says that the label is 1,
		#otherwise, go down to the next highest confidence classifier.  If that label is -1, 
		#then classify the tweet as the third highest confidence label.  
		#(e.g. if its confidently not love, confidently not hate, then say it is sadness)
		if temp[order[i][2]] == 1:
			prediction.append(order[i][2])
		elif temp[order[i][1]] == 1:
			prediction.append(order[i][1])
		else:
			prediction.append(order[i][0])


	#match 0,1,2 to love, hate, and sadness
	for i in range(0,x):
		if prediction[i] == 0:
			prediction[i] = "love"
		elif prediction[i] == 1:
			prediction[i] = "hate"
		else:
			prediction[i] = "sadness"

	return prediction


def main(): #ONE VS ALL TERNARY CLASSIFIER
	dataframe = load_data("training_data.csv")
	test = load_data("test_data.csv")

	#get the feature matrix (num_tweets, dictionary_length) of the training set
	#and a dictionary of words that are represented by the feature vector for each individual data point.
	fm, dictionary = initialize_stuff(dataframe)

	#get three binary classifiers, each optimized to the best hyperparameter (using cross validation)
	#each classifier considers one class to be positive, and everything else to be negative (one vs all) 
	love,hate,sadness = generate_classifiers(fm,dataframe)

	#generate feature matrix for test data set, similar to what initialize_stuff(dataframe) does on training set
	test_fm = np.zeros((test['content'].size, 1360))
	tweet_number = 0
	for tweet in test['content']:
		#replace punctuation with whitespace, removes numbers and apostrophe's without replacing with whitespace
	  	transtable = str.maketrans(".,?!\";:#$%^&*()-=+_/\|[]{}<>", "                            ")
	  	tweet = tweet.translate(transtable)
	  	transtable = str.maketrans("","", "'1234567890")
	  	tweet = tweet.translate(transtable).lower()

	  	#no need to add new words to dictionary, as this is the test set
	  	#set the feature matrix depending on the number of times each word in the dictionary appears in the tweet
  		for key, value in dictionary.items():
	  		if key in tweet.split():
	  			test_fm[tweet_number][value] = tweet.split().count(key)
	  	tweet_number += 1

	#Calculate my classifier's predictions on test_data.csv by selecting the highest confidence classifier
	predictions = calculate_prediction(love, hate, sadness, test_fm)
	pd.Series(np.array(predictions)).to_csv('MyPredictions.csv', index=False)
	print("\nPredictions saved to \"MyPredictions.csv\"")

	#See how well your classifier classifies the data that it was trained on. 
	#100% means you are overfitting!!!
	training_predictions = calculate_prediction(love,hate,sadness,fm)
	print("\nConfusion matrix for your classifier on the original training data:\n")
	print(confusion_matrix(dataframe['sentiment'], training_predictions, labels = ["love", "hate", "sadness"]))

	return


if __name__ == '__main__':
  	main()