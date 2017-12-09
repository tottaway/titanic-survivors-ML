import numpy as np
import csv
from helpers import formatAttributes, dictToNDArray, featureNorm, logisticRegression,test


"""
TODO: FEATURE SCALING
TODO: find a good minimizing function for linear regression
TODO: set up logistic regression cost function
TODO: figure how to deal with decimals better
"""

# opens csv and pulls out data
with open('train.csv') as csvfile:
    myCsv = csv.reader(csvfile, delimiter=',')
    passengerDict = {}
    testDict = {}
    myCsv.__next__()
    for n in range(1, 600):
        passengerDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))
    for n in range(1, 192):
        testDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))

Xtest, Ytest = dictToNDArray(passengerDict)
X, y = dictToNDArray(passengerDict)
theta = logisticRegression(X, y)[0]
print(test(Xtest, Ytest, theta))


