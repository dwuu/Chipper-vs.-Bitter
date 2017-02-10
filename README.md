# Chipper-vs.-Bitter
Using a training set of text-based tweet and their labels (love, hate, or sadness), I designed a linear classifier that attempts to best classify new data.  Because this is a ternary classification problem, I chose to use a one v. all approach to simplify it to a binary problem.

ChipperVsBitter.py is the python program. 
It reads the training_data.csv and test_data.csv.

Training data is presented as:

  sentiment, content
  love, I love my dog so much.

It outputs predicted labels for the test data to MyPredictions.csv using the linear classifier I built.

The gist of my program:

1) Create a dictionary (length d) of words extracted from n tweets from the training set.  Ignore words that begin with "@" (for mentions), and ignore punctuation and numbers.

2) For each tweet, count how many times each word in the dictionary is found in that tweet.  The feature vector for each tweet is therefore a vector of length d, where the values are equal to the number of times each word appears.  The feature matrix X is a n * d matrix that represents every tweet in the training set.

3) Create 3 classifiers, one for each of love, hate, and sadness.  For each of the these classifiers:
    -It is essentially a binary classifier that calls its sentiment 1 and everything else -1. 
    -Test their performance (based on the AUROC metric) on several magnitudes of hyperparameters
    -Performance is checked using stratified k-fold cross validation (each split has the same ratio of love, hate, and sad tweets)
    -Pick the best hyperparameter C for each classifier, and train each classifier on the training set with their respective optimal C
    
4) Similar to step 2, create the feature matrix for the test set.  Each feature vector is of length d.

5) Using the 3 optimized classifiers, we now must decide which one to go with when deciding our final prediction.
    -For each new piece of test data, check the 3 classifiers in order of highest confidence.  
      -If the highest confidence classifier (for sentiment A) classifies it as 1, label the tweet A.  
      -Otherwise, look at the next highest confidence classifier (sentiment B).  If that is 1, label the tweet B.
      -Otherwise, label the tweet the last sentiment.
      
6) These predictions are then output to MyPredictions.csv

7) My program also outputs a confusion matrix for the classifier's performance on the original training data.
    -Why is this useful?  Shouldn't your classifier have a 100% classification rate on the data that it was trained on?
      -NO.  If it is perfect, then your classifier is likely to be overfit.  This confusion matrix thus should show that we have some errors, which it does:

[[100   0   0]
 [  0 100   0]
 [  0   8  92]]

NOTES: I considered several different models, including linear and higher-dimensional kernels, L1 and L2 regularization, squared--hinge-loss and hinge-loss, etc.  I found that the Linear-kernel SVM with normal hinge-loss and L2 penalty worked the best for this particular classification problem.  This was optimized through trial and error.

Earlier in the project I also experimented with different measures of optimality regarding the training set's performance, by calculating confidence intervals using a bootstrapping sampling method.  While this may be useful in some contexts, it was not crucial to the optimization of my classifier, and I just used the stratified k-folds CV to measure predictive performance.
