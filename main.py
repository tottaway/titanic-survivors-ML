import numpy
import csv
from helpers import formatAttributes, dictToNDArray, featureNorm

np = numpy
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
    myCsv.__next__()
    for n in range(1, 892):
        passengerDict[n] = featureNorm(formatAttributes(myCsv.__next__()[1:12]))


X, y = dictToNDArray(passengerDict)

