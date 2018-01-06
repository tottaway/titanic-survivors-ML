import numpy as np

def backprop(a, z, nnParams, grad, X, y, m, alpha, s):
    l = len(grad)
    dz = {}
    dz[l] = a[l] - y
    grad[l] = (1 / m) * np.dot(dz[l], np.transpose(a[l-1]))

    for n in range(1, l):
        layer = l - n
        dz[layer] = np.dot(np.transpose(nnParams[layer+1]), dz[layer+1]) * sigmoidDerivative(z[layer])
        grad[layer] = (1 / m) * np.dot(dz[layer], np.transpose(a[layer-1])) + ((s / m) * nnParams[layer])
    for n in range(1, len(nnParams) + 1):
        nnParams[n] -= (alpha * grad[n])

    return nnParams


def dictToNDArray(d, t=False):
    """
    takes passangerDict and returns an ndarray
    :param d: dict
    :return: if t=True a ndarray; else a tuple of ndarrays
    """
    if not t:
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


def debugShapes(a, z, nnParams, grad):
    shapes = [["a: "], ["z: "], ["nnParams: "], ["grad: "]]
    l = [a, z, nnParams, grad]
    for i in range(len(l)):
        for n in l[i].values():
            shapes[i].append(n.shape)

    print(shapes)
    pass


def featureNorm(X):
    """
    scales data so norm is 1
    :param l: ndarray
    :return: ndarray
    """
    XNorm = np.linalg.norm(X, axis=1, keepdims=True)
    X /= XNorm
    return X


def feedforward(a, z, nnParams):
    for n in range(1, len(a)):
        z[n] = np.dot(nnParams[n], a[n-1])
        a[n] = sigmoid(z[n])
    return a, z


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

    # formats sex and removes NaN data
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

    l[9] = formatCabin(l[9])

    # drop ticket numbers or names
    del l[2]
    del l[6]

    return formatRemainder(l)


def initNN(layerSizes, numHidenLayers, X):
    a = {}
    z = {}
    nnParams = {}
    grad = {}
    a[0] = X
    for n in range(1, (numHidenLayers + 2)):
        size = (layerSizes[n], X.shape[1])
        a[n], z[n] = np.ones(size), np.ones(size)

    nnParams[1] = np.random.rand(layerSizes[1], layerSizes[0])
    grad[1] = np.zeros((layerSizes[1], layerSizes[0]))
    for n in range(2, numHidenLayers + 2):
        if n < (numHidenLayers + 1):
            nnParams[n] = np.random.rand(layerSizes[n], layerSizes[n-1])
            grad[n] = np.zeros((layerSizes[n], layerSizes[n-1]))
        else:
            nnParams[n] = np.random.rand(layerSizes[-1], layerSizes[-2])
            grad[n] = np.zeros((layerSizes[-1], layerSizes[-2]))

    return a, z, grad, nnParams


def logisticRegression(X, y, alpha=0.05, s=0, poly=0):
    """
    Runs logistic regression on the training data outputting weights as well as the cost

    :param X: training data (ndarray)
    :param y: labels for training data (ndarray)
    :param alpha: learning rate (num)
    :param s: regularization constant (num)
    :param poly: creates new features (num)
    :return: weights and cost (tuple of an ndarray and a num)
    """

    # create new features based off of powers of current features
    # allows the model to fit more non-linear data
    if poly != 0:
        polyFeat = np.zeros((X.shape[0], (poly * 10)))
        for n in range(2, poly + 2):
            index = range(((n-2) * 10), ((n-2) * 10 + 10))
            polyFeat[:, index] = np.power(X[poly - 2], n)
        np.append(X, polyFeat)

    np.append(X, np.ones(X.shape[0]))

    # initialize variables
    # m := num of training examples
    # J := cost
    # count := used to limit runtime
    # theta := weights
    m = y.shape[0]
    theta = np.random.rand((X.shape[1]), 1)
    count = 0
    J = 1

    def costFunction(X, y, theta, m, s=0):
        """
        :return: num
        """
        # hypothesis
        h = sigmoid(np.dot(X, theta))

        # calculate cost for each example
        temp = (1/m) * ((np.dot(np.transpose(np.negative(y)), np.log(h))) - np.dot((1-np.transpose(y)), np.log(1-h)))
        # total cost
        temp = temp.sum((0, 1))
        # regularize
        temp += (s / (2 * m)) * np.sum(theta[:-1])
        return temp

    def grad(X, y, theta, m, alpha, s=0):
        """
        Calculates the partial derivatives of the cost function
        :return: ndarry
        """
        # hypothesis
        h = sigmoid(np.dot(X, theta))

        theta -= (alpha / m) * (np.dot(np.transpose(X), (h - y)))
        # regularization
        theta[:-1] -= ((s / m) * theta[:-1])
        return theta

    # run gradient descent until cost drops below a given threshold or the function times out
    while J > 0.001 and count < 100000:
        # update theta
        theta = grad(X, y, theta, m, alpha, s)
        # calculate cost every 1000 iterations
        if count % 1000 == 0:
            J = costFunction(X, y, theta, m, s)
            #print(J)

        count += 1

    return theta, costFunction(X, y, theta, m, s)


def neuralNet(hidenLayerSizes, X, y, s=0, g=0, alpha=0.05):
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
    # init variables
    layerSizes = [X.shape[0]]
    layerSizes.extend(hidenLayerSizes)
    layerSizes.append(y.shape[0])
    numHidenLayers = len(hidenLayerSizes)
    m = X.shape[1]
    a, z, grad, nnParams = initNN(layerSizes, numHidenLayers, X)

    # prints shapes of various objects for debuggin purposes
    debugShapes(a, z, nnParams, grad)

    # run gradient descent
    count = 0
    while count <= 100000:
        a, z = feedforward(a, z, nnParams)
        nnParams = backprop(a, z, nnParams, grad, X, y, m, alpha, s)
        if count % 5000 == 0:
            if count == 0:
                print("Iteration: " + str(count//5000))
            else:
                print(str(count//5000))

        count += 1

    accuracy = testNN(hidenLayerSizes, X, y, nnParams)
    return nnParams, accuracy


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


def sigmoidDerivative(X, L=1, K=1, x0=0):
    """

    :param X: ndarray
    :return: ndarray
    """
    s = sigmoid(X, L, K, x0)
    return s * (1 - s)


def test(Xtest, y, theta):
    """
    predicts y based on theta then calculates percent correct
    :param Xtest: ndarray
    :param y: ndarray
    :param theta: ndarray
    :return: float
    """
    h = sigmoid(np.dot(featureNorm(Xtest), theta))
    for n in h:
        if n <= 0.5:
            n = 0
        else:
            n = 1

    score = np.abs((y - h))
    return 100 - (np.mean(score) * 100)


def testNN(hidenLayerSizes, Xtest, y, nnParams):
    def initTestNN(layerSizes, numHidenLayers, X):
        a = {}
        z = {}
        a[0] = Xtest
        for n in range(1, (numHidenLayers + 2)):
            size = (layerSizes[n], Xtest.shape[1])
            a[n], z[n] = np.ones(size), np.ones(size)

        return a, z

    # init variables
    layerSizes = [Xtest.shape[0]]
    layerSizes.extend(hidenLayerSizes)
    layerSizes.append(y.shape[0])
    numLayers = len(layerSizes)
    numHidenLayers = len(hidenLayerSizes)
    a, z = initTestNN(layerSizes, numHidenLayers, Xtest)

    # calculate hypothesis
    a, z = feedforward(a, z, nnParams)
    h = np.squeeze(np.round(np.transpose(a[numLayers-1])))
    y = np.squeeze(y)

    score = np.abs((y - h))
    return (np.mean(score) * 100)