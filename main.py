import matplotlib.pyplot as plt
import numpy as np
import csv
from helpers import formatAttributes, dictToNDArray, featureNorm, logisticRegression,test


"""
TODO: submit to kaggle
TODO: find a good minimizing function for logistic regression
TODO: fine tune logistic regression
TODO: figure how to deal with decimals better
TODO: set up neural net
"""

# opens csv and pulls out data
with open('train.csv') as csvfile:
    myCsv = csv.reader(csvfile, delimiter=',')
    testCsv = csv.reader(csvfile, delimiter=',')
    passengerDict = {}
    testDict = {}
    myCsv.__next__()
    n = 0
    while True:
        try:
            passengerDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))
            testDict[n] = featureNorm(formatAttributes(testCsv.__next__()[1:12]))
            n += 1
        except:
            break

Xtest = dictToNDArray(testDict, t=True)
X, y = dictToNDArray(passengerDict, t=False)
theta, cost = logisticRegression(X, y, s=7, poly=2)

print(test(Xtest, Ytest, theta), theta)