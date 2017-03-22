"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
from boostedDT import BoostedDT

print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from boostedDT import BoostedDT

# load the data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

n,d = X.shape
nTrain = 0.5*n  #training on 50% of the data

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

# train the decision tree
#modelDT = DecisionTreeClassifier()
#modelDT.fit(Xtrain,ytrain)

# train the boosted DT
modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
modelBoostedDT.fit(Xtrain,ytrain)

# output predictions on the remaining data
#ypred_DT = modelDT.predict(Xtest)
ypred_BoostedDT = modelBoostedDT.predict(Xtest)

# compute the training accuracy of the model
#accuracyDT = accuracy_score(ytest, ypred_DT)
accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)

#print "Decision Tree Accuracy = "+str(accuracyDT)
print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)

# predict data/challengeTrainLabeled.dat
newBoosted = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
trainDataset = np.loadtxt('data/challengeTrainLabeled.dat', delimiter=',')
n,d = trainDataset.shape
Xtrain = trainDataset[:,:d-1]
ytrain = trainDataset[:,d-1]
testDataset = np.loadtxt('data/challengeTestUnlabeled.dat', delimiter=',')
newBoosted.fit(Xtrain, ytrain)
train_pred = newBoosted.predict(Xtrain)
accuracyChallenge = accuracy_score(ytrain, train_pred)
print "accuracy for the challenge training data set is " + str(accuracyChallenge)
challengePred = newBoosted.predict(testDataset)
predictions = ""
for i in range(0, challengePred.size):
    predictions += str(int(challengePred[i])) + ","  
predictions = predictions[:-1]
print predictions


kNeighbors = neighbors.KNeighborsClassifier()
kNeighbors.fit(Xtrain, ytrain)
kNeighbors_pred = kNeighbors.predict(Xtrain)
accuracykNN = accuracy_score(ytrain, kNeighbors_pred)
print "accuracy for the kNN training is " + str(accuracykNN)
challenge_kNN_pred = kNeighbors.predict(testDataset)
predictions = ""
for i in range(0, challenge_kNN_pred.size):
    predictions += str(int(challenge_kNN_pred[i])) + ","  
predictions = predictions[:-1]
print predictions