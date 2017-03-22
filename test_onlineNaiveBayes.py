"""
==============================================================
Test the online naive Bayes against the standard naive Bayes
==============================================================

Author: Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from naiveBayes import NaiveBayes
from naiveBayes import OnlineNaiveBayes

# load the data set
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

n,d = X.shape
nTrain = int(0.5*n)  #training on 50% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# train the online naive Bayes
modelONB = OnlineNaiveBayes(useLaplaceSmoothing=True)
for i in range(nTrain):
    modelONB.fit(Xtrain[i:i+1,:],ytrain[i:i+1])  # train two instances at a time

# train the boosted ONB
modelNB = NaiveBayes(useLaplaceSmoothing=True)
modelNB.fit(Xtrain,ytrain)

# output predictions on the remaining data
ypred_ONB = modelONB.predict(Xtest)
ypred_NB = modelNB.predict(Xtest)

# compute the training accuracy of the model
accuracyONB = accuracy_score(ytest, ypred_ONB)
accuracyNB = accuracy_score(ytest, ypred_NB)

print "Online Naive Bayes Accuracy = "+str(accuracyONB)
print "Batch Naive Bayes Accuracy = "+str(accuracyNB)
