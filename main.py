import matplotlib.pyplot as plt
import numpy as np
import csv
from helpers import format_attributes, dict_to_ndarray, feature_norm, logistic_regression, test, neural_net, test_nn


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
    my_csv = csv.reader(csvfile, delimiter=',')
    test_csv = csv.reader(csvfile, delimiter=',')
    train_dict = {}
    dev_dict = {}
    test_dict = {}
    my_csv.__next__()
    i, j = 0, 0
    while True:
        try:
            for n in range(4):
                train_dict[i] = format_attributes(my_csv.__next__()[1:12])
                i += 1
            dev_dict[j] = format_attributes(my_csv.__next__()[1:12])
            j += 1
        except:
            break


    # while True:
    #     try:
    #         test_dict[n] = feature_norm(format_attributes(test_csv.__next__()[1:12]))
    #         n += 1
    #     except:
    #         break


# x_test = dict_to_ndarray(test_dict, t=True)
x_dev, y_dev = dict_to_ndarray(dev_dict, t=False)
x, y = dict_to_ndarray(train_dict, t=False)
x, x_dev = feature_norm(x), feature_norm(x_dev)


# transposing things because I set them up backwards
x, y, x_dev, y_dev = np.transpose(x), np.transpose(y), np.transpose(x_dev), np.transpose(y_dev)
# describe NN structure
hiden_layer_sizes = (10, 5)
nn_params, training_accuracy = neural_net(hiden_layer_sizes, x, y, s=0.3)
# theta, cost = logisticRegression(x, y, s=-1, poly=0)

print("Dev set error: " + str(test_nn(hiden_layer_sizes, x_dev, y_dev, nn_params)) + "%")
print("Training set error: " + str(training_accuracy) + "%")
