'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.no_of_boosting_iters = numBoostingIters
        self.max_tree_depth = maxTreeDepth
        self.total_no_of_classes = None
        self.classes = None
        self.clf = [None]*numBoostingIters
        self.beta = None
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
        # assign equal weights to all instances initially
	n = X.shape[0]
	#training_error = 0
	weight_of_instances = np.ones(n)/float(n)
	self.total_no_of_classes = np.unique(y).size 
	self.classes = np.unique(y)		
	self.beta = np.zeros(self.no_of_boosting_iters)
        
        for i in xrange(self.no_of_boosting_iters):
            self.clf[i] = tree.DecisionTreeClassifier(max_depth = self.max_tree_depth)
            self.clf[i] = self.clf[i].fit(X,y,sample_weight = weight_of_instances)
	    predictions = self.clf[i].predict(X)
	    training_error = 0
	    
	    for j in xrange(n):
 		    if y[j] != predictions[j]:
	  	        training_error += weight_of_instances[j]
	    betas = ((np.log((1 - training_error) / training_error) + np.log(self.total_no_of_classes - 1)) * 1 / 2)
	    self.beta[i] = betas
          
	    for k in xrange(n):
		flag = 0
		if predictions[k] == y[k]:
		    flag = 1
		if flag != 0:
		    weight_of_instances[k] *= np.exp(betas*(-1))
            normalized = np.sum(weight_of_instances)

	    for l in xrange(n):
	        weight_of_instances[l] /= normalized
          
    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        beta = np.zeros((X.shape[0], self.total_no_of_classes))
        for iteration in xrange(self.no_of_boosting_iters):
            predictions = self.clf[iteration].predict(X)
            for classes in xrange(self.total_no_of_classes):
                beta[:, classes] += (predictions == self.classes[classes]) * self.beta[iteration]
        return self.classes[np.argmax(beta, axis = 1)]
