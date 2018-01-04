import matplotlib.pyplot as plt
import numpy as np
import csv
from helpers import formatAttributes, dictToNDArray, featureNorm, logisticRegression, test, neuralNet


"""
My model is acting really weird. The system appears to become more accurate on the cross validation set as the values
of theta grow. This means that a positive regularization constant reduces accuracy while a negative regularization
constant increases accuracy. No idea why this is happening.

TODO: submit to kaggle
TODO: fine tune logistic regression
TODO: figure how to deal with decimals better
TODO: set up neural net
"""

# opens csv and pulls out data
with open('train.csv') as csvfile:
    myCsv = csv.reader(csvfile, delimiter=',')
    testCsv = csv.reader(csvfile, delimiter=',')
    trainDict = {}
    devDict = {}
    testDict = {}
    myCsv.__next__()
    n = 0
    for n in range(1, 600):
        trainDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))
    for n in range(1, 192):
        devDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))

    # while True:
    #     try:
    #         testDict[n] = featureNorm(formatAttributes(testCsv.__next__()[1:12]))
    #         n += 1
    #     except:
    #         break

#print(neuralNet(10, [100, 25], 10, 2, np.zeros(10)), np.zeros([0]), 0)

#Xtest = dictToNDArray(testDict, t=True)
Xdev, Ydev = dictToNDArray(devDict, t=False)
X, y = dictToNDArray(trainDict, t=False)
theta, cost = logisticRegression(X, y, s=0, poly=0)

print(test(Xdev, Ydev, theta), theta)