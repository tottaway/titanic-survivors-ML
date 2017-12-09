import numpy as np


def dictToNDArray(d):
    """
    takes passangerDict and returns an ndarray
    :param d: dict
    :return: tuple Aof ndarrays
    """
    X = []
    y = []
    for i in d.keys():
        X += [d[i][1:11]]
        y += [[d[i][0]]]

    X, y = np.array(X), np.array(y)
    return (X, y)


def featureNorm(l):
    """
    scales data so that max is around 1, min is around -1 and mean is around 0
    :param l: list
    :return: list
    """
    l[1] = (l[1] - 2)
    l[3] = (l[3] - 0.5) * 2
    l[4] = (l[4] - 23) / 35
    l[5] = l[5] / 5
    l[6] = l[6] / 6
    l[8] = (l[8] - 30) / 500
    l[9] = (l[9] - 50) / 600
    l[10] = (l[10] - 0.3) / 3
    return l




def formatAttributes(l):
    """
    removes strings and turns letters into numbers
    :param l: list
    :return: list
    """
    # Takes letters in cabin number replaces them w/ nums
    def formatCabin(cabin):
        letterToNum = {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6'}
        for key in letterToNum.keys():
            if key in cabin:
                cabin = cabin.replace(key, letterToNum[key])[-3:]
        return cabin

    # formats sex and embarked data points
    def formatRemainder(l):
        conversionDict = {'male': 0, 'female': 1, 'S': 0, 'C': 1, 'Q': 2,}
        result = []
        for n in l:
            if n in conversionDict.keys():
                n = conversionDict[n]

            # strips remaining letters
            try:
                if n.count('.') > 1:
                    n = "".join(_ for _ in n if _ in "1234567890")
                else:
                    n = "".join(_ for _ in n if _ in ".1234567890")
            except:
                pass

            # takes out null data points
            if n == None or n == '' or n == '.':
                n = 0

            # str to num
            n = float(n)
            result = result + [n]
        return result

    # I don't know how to deal with ticket numbers or names
    l[2], l[7] = 0, 0

    l[9] = formatCabin(l[9])
    return formatRemainder(l)


def logisticRegression(X, y, alpha=0.05, s=7):
    m = y.shape[0]
    theta = np.random.rand(X.shape[1], 1)
    count = 0
    J = 11

    def costFunction(X, y, theta, m, s=0):
        h = sigmoid(np.dot(X, theta))
        temp = (1/m) * ((np.dot(np.transpose(np.negative(y)), np.log(h))) - np.dot((1 - np.transpose(y)), np.log(1 - h)))
        temp = temp.sum((0,1))
        temp += (s / (2 * m)) * np.sum(theta)
        return temp

    def grad(X, y, theta, m, alpha, s=0):
        temp = theta - (alpha / m) * (np.dot(np.transpose(X), (sigmoid(np.dot(X, theta)) - y)))
        return temp + ((s / m) * theta)
    while J > 0.001 and count < 150000:
        theta = grad(X, y, theta, m, alpha, s)
        J = costFunction(X, y, theta, m, s)
        count += 1

    return theta, costFunction(X, y, theta, m, s)


def sigmoid(X, L=1, k=1,x0=0):
    """
    Preforms sigmoid function element wise on X

    L, k, x0 are constants
    :param X: ndarray
    :param L: num
    :param k: num
    :param x0: num
    :return: ndarray
    """
    return L/(1 + (np.exp((np.negative(k*(X - x0))))))


def test(Xtest, y, theta):
    h = sigmoid(np.dot(Xtest, theta))
    for n in h:
        if n <= 0.5:
            n = 0
        else:
            n = 1

    score = np.abs((y - h))
    return 100 - np.mean(score) * 100
