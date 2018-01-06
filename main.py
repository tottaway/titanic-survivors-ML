import matplotlib.pyplot as plt
import numpy as np
import csv
from helpers import formatAttributes, dictToNDArray, featureNorm, logisticRegression, test, neuralNet, testNN


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
    i, j = 0, 0
    while True:
        try:
            for n in range(4):
                trainDict[i] = formatAttributes(myCsv.__next__()[1:12])
                i += 1
            devDict[j] = formatAttributes(myCsv.__next__()[1:12])
            j += 1
        except:
            break


    # while True:
    #     try:
    #         testDict[n] = featureNorm(formatAttributes(testCsv.__next__()[1:12]))
    #         n += 1
    #     except:
    #         break


#Xtest = dictToNDArray(testDict, t=True)
Xdev, Ydev = dictToNDArray(devDict, t=False)
X, y = dictToNDArray(trainDict, t=False)
X, Xdev = featureNorm(X), featureNorm(Xdev)


# transposing things because I set them up backwards
X, y, Xdev, Ydev = np.transpose(X), np.transpose(y), np.transpose(Xdev), np.transpose(Ydev)
# describe NN structure
hidenLayerSizes = (10, 5)
nnParams, trainingAccuracy = neuralNet(hidenLayerSizes, X, y, s=1)
#theta, cost = logisticRegression(X, y, s=-1, poly=0)

print("Dev set error: " + str(testNN(hidenLayerSizes, Xdev, Ydev, nnParams)) + "%")
print("Training set error: " + str(trainingAccuracy) + "%")
