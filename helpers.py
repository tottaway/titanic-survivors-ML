import numpy as np


def dictToNDArray(d, t=False):
    """
    takes passangerDict and returns an ndarray
    :param d: dict
    :return: if t=True a ndarray; else a tuple of ndarrays
    """
    if not test:
        X = []
        y = []
        for i in d.keys():
            X += [d[i][1:11]]
            y += [[d[i][0]]]
        X = np.array(X)
        y = np.array(y)
        return X, y
    else:
        X = []
        for i in d.keys():
            X += [d[i][0:10]]
        X =np.array(X)
        return X


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


def logisticRegression(X, y, alpha=0.05, s=0, poly=0):
    if poly != 0:
        polyFeat = np.zeros((X.shape[0], (poly * 10)))
        for n in range(2, poly + 2):
            index = range(((n-2) * 10), ((n-2) * 10 + 10))
            polyFeat[:, index] = np.power(X[poly - 2], n)
        np.append(X, polyFeat)

    m = y.shape[0]
    theta = np.random.rand(X.shape[1], 1)
    count = 0
    J = 11

    def costFunction(X, y, theta, m, s=0):
        h = sigmoid(np.dot(X, theta))
        temp = (1/m) * ((np.dot(np.transpose(np.negative(y)), np.log(h))) - np.dot((1-np.transpose(y)), np.log(1-h)))
        temp = temp.sum((0, 1))
        temp += (s / (2 * m)) * np.sum(theta)
        return temp

    def grad(X, y, theta, m, alpha, s=0):
        temp = theta - (alpha / m) * (np.dot(np.transpose(X), (sigmoid(np.dot(X, theta)) - y)))
        return temp + ((s / m) * theta)

    while J > 0.001 and count < 100000:
        theta = grad(X, y, theta, m, alpha, s)
        J = costFunction(X, y, theta, m, s)
        count += 1

    return theta, costFunction(X, y, theta, m, s)


def neuralNet(inputLayerSize, hidenLayerSizes, numLabels, numHidenLayers, X, y, s):
    """

    :param inputLayerSize: int
    :param hidenLayerSizes: tuple of ints
    :param numLabels: int
    :param numHidenLayers: int
    :param X: array of data, size = (1, inputLayerSize)
    :param y: vector
    :param s: int: lambda
    :return: tuple (cost, gradients)
    """
    assert X.shape == (1, inputLayerSize), "X.shape must equal (1, inputLayerSize)"
    def initParams(inputLayerSize, hidenLayerSizes, numHidenLayers, outputLayerSize):
        nnParams = {}
        nnParams[0] = np.random.rand(inputLayerSize, hidenLayerSizes[0])
        for n in range(0, (numHidenLayers)):
            if n < (numHidenLayers - 1):
                nnParams[n + 1] = np.random.rand(hidenLayerSizes[n], hidenLayerSizes[n+1])
            else:
                nnParams[n + 1] = np.random.rand(hidenLayerSizes[n], outputLayerSize)
    pass


def reformat(initialData)

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
