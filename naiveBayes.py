'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn.preprocessing import normalize

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.totalClasses = None
        self.conditionalProb = None
        self.classProb = None
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n, d = X.shape
        self.totalClasses = np.unique(y)
        total_no_of_classes = np.unique(y).size
        s = [total_no_of_classes, d]
        self.conditionalProb = np.zeros(s)
        self.classProb = np.zeros(total_no_of_classes)
        
        if (self.useLaplaceSmoothing == False):
            for cls in xrange(total_no_of_classes):
                cle = X[np.logical_or.reduce([y == self.totalClasses[cls]])]
                cSum = np.sum(cle, axis = 0)
                dSum = np.sum(cle)
                self.conditionalProb[cls, :] = cSum / dSum
                clshape = cle.shape[0]
                self.classProb[cls] = clshape / float(n)
        else:
            for cls in xrange(total_no_of_classes):
                cle = X[np.logical_or.reduce([y == self.totalClasses[cls]])]
                cSum = np.sum(cle, axis = 0)
                dSum = np.sum(cle)
                self.conditionalProb[cls, :] = (cSum + 1.0) / (d + dSum)
                clshape = cle.shape[0]
                self.classProb[cls] = clshape / float(n)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        predictions = self.predictProbs(X)
        sign = np.argmax(predictions, axis = 1)
        return self.totalClasses[sign]
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        predictions = np.matrix(X, copy=False)*np.matrix(np.log(self.conditionalProb).T, copy=False)
        predictions = predictions + np.log(self.classProb)
        predictions = predictions - np.mean(predictions)
        predictions = np.exp(predictions)
        return normalize(predictions, norm = 'l1', axis = 1)
        
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.totalClasses = None
        self.conditionalProb = None
        self.classProb = None
        self.noOfClasses = None
        self.noOfFeatures = None
        self.sum = None
        self.rows = None      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n, d = X.shape
        if self.rows != None:
            self.rows = self.rows + n
        else:
            self.rows = n
        if self.totalClasses == None:
            ind = 0
            self.totalClasses = np.unique(y)
        else:
            totalClasses = np.unique(np.r_[self.totalClasses, y])
            a = totalClasses.size
            b = self.totalClasses.size
            ind = a - b
            if (ind > 0):
                checkClasses = np.in1d(totalClasses, self.totalClasses, assume_unique = False, invert = False)
                test = np.logical_or.reduce([checkClasses == False])
                testClasses = totalClasses[test]
                self.totalClasses = np.concatenate((self.totalClasses, testClasses), axis = 0)
        b = self.totalClasses.size
        if self.conditionalProb == None:
            self.conditionalProb = np.zeros([b, d])
        if self.classProb == None:
            self.classProb = np.zeros(b)
        if self.noOfClasses == None:
            self.noOfClasses = np.zeros(b)
        if self.noOfFeatures == None:
            self.noOfFeatures = np.zeros([b, d])
        if self.sum == None:
            self.sum = np.zeros(b)
        
        if (ind > 0):
            c = np.zeros([ind, d])
            e = np.zeros(ind)
            self.conditionalProb = np.concatenate((self.conditionalProb, c), axis = 0)
            self.classProb = np.concatenate((self.classProb, e), axis = 0)
            self.noOfClasses = np.concatenate((self.noOfClasses, e), axis = 0)
            self.noOfFeatures = np.concatenate((self.noOfFeatures, c), axis = 0)
            self.sum = np.concatenate((self.sum, e), axis = 0)
            
        if (self.useLaplaceSmoothing == False):
            for cls in xrange(b):
                cle = X[np.logical_or.reduce([y == self.totalClasses[cls]])]
                cSum = np.sum(cle, axis = 0)
                dSum = np.sum(cle)
                self.conditionalProb[cls, :] = cSum / dSum
                clshape = cle.shape[0]
                self.classProb[cls] = clshape / float(n)
        else:
            for cls in xrange(b):
                cle = X[np.logical_or.reduce([y == self.totalClasses[cls]])]
                if (cle.shape[0] > 0):
                    self.noOfClasses[cls] = self.noOfClasses[cls] + cle.shape[0]
                    self.classProb[cls] = self.noOfClasses[cls]/float(self.rows)
                    self.noOfFeatures[cls, :] = self.noOfFeatures[cls, :] + np.sum(cle, axis = 0)
                    self.sum[cls] = self.sum[cls] + np.sum(cle)
                    self.conditionalProb[cls, :] = (1.0 + self.noOfFeatures[cls, :])/(d + self.sum[cls])


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        predictions = self.predictProbs(X)
        sign = np.argmax(predictions, axis = 1)
        finalPredictions = self.totalClasses[np.argsort(self.totalClasses)]
        return finalPredictions[sign]

        
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        predictions = np.matrix(X, copy=False)*np.matrix(np.log(self.conditionalProb).T, copy=False)
        predictions = predictions + np.log(self.classProb)
        predictions = predictions - np.mean(predictions)
        predictions = np.exp(predictions)
        normalized_l1 = normalize(predictions, norm = 'l1', axis = 1)
        sortClasses = np.argsort(self.totalClasses)
        return normalized_l1[:, sortClasses]
