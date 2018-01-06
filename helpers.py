import numpy as np

def backprop(a, z, nn_params, grad, x, y, m, alpha, s):
    l = len(grad)
    dz = {}
    dz[l] = a[l] - y
    grad[l] = (1 / m) * np.dot(dz[l], np.transpose(a[l-1]))

    for n in range(1, l):
        layer = l - n
        dz[layer] = np.dot(np.transpose(nn_params[layer+1]), dz[layer+1]) * sigmoid_derivative(z[layer])
        grad[layer] = (1 / m) * np.dot(dz[layer], np.transpose(a[layer-1])) + ((s / m) * nn_params[layer])
    for n in range(1, len(nn_params) + 1):
        nn_params[n] -= (alpha * grad[n])

    return nn_params


def dict_to_ndarray(d, t=False):
    """
    takes passangerDict and returns an ndarray
    :param d: dict
    :param t: bool (test)
    :return: if t=True a ndarray; else a tuple of ndarrays
    """
    if not t:
        x = []
        y = []
        for i in d.keys():
            x += [d[i][1:11]]
            y += [[d[i][0]]]
        x = np.array(x)
        y = np.array(y)
        return x, y
    else:
        x = []
        for i in d.keys():
            x += [d[i][0:10]]
        x =np.array(x)
        return x


def debug_shapes(a, z, nn_params, grad):
    shapes = [["a: "], ["z: "], ["nn_params: "], ["grad: "]]
    l = [a, z, nn_params, grad]
    for i in range(len(l)):
        for n in l[i].values():
            shapes[i].append(n.shape)

    print(shapes)
    pass


def feature_norm(x):
    """
    scales data so norm is 1
    :param l: ndarray
    :return: ndarray
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= x_norm
    return x


def feedforward(a, z, nn_params):
    for n in range(1, len(a)):
        z[n] = np.dot(nn_params[n], a[n-1])
        a[n] = sigmoid(z[n])
    return a, z


def format_attributes(l):
    """
    removes strings and turns letters into numbers
    :param l: list
    :return: list
    """
    # Takes letters in cabin number replaces them w/ nums
    def format_cabin(cabin):
        letter_to_num = {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6'}
        for key in letter_to_num.keys():
            if key in cabin:
                cabin = cabin.replace(key, letter_to_num[key])[-3:]
        return cabin

    # formats sex and removes NaN data
    def format_remainder(l):
        conversion_dict = {'male': 0, 'female': 1, 'S': 0, 'C': 1, 'Q': 2,}
        result = []
        for n in l:
            if n in conversion_dict.keys():
                n = conversion_dict[n]

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

    l[9] = format_cabin(l[9])

    # drop ticket numbers or names
    del l[2]
    del l[6]

    return format_remainder(l)


def init_nn(layer_sizes, num_hiden_layers, x):
    a = {}
    z = {}
    nn_params = {}
    grad = {}
    a[0] = x
    for n in range(1, (num_hiden_layers + 2)):
        size = (layer_sizes[n], x.shape[1])
        a[n], z[n] = np.ones(size), np.ones(size)

    nn_params[1] = np.random.rand(layer_sizes[1], layer_sizes[0])
    grad[1] = np.zeros((layer_sizes[1], layer_sizes[0]))
    for n in range(2, num_hiden_layers + 2):
        if n < (num_hiden_layers + 1):
            nn_params[n] = np.random.rand(layer_sizes[n], layer_sizes[n-1])
            grad[n] = np.zeros((layer_sizes[n], layer_sizes[n-1]))
        else:
            nn_params[n] = np.random.rand(layer_sizes[-1], layer_sizes[-2])
            grad[n] = np.zeros((layer_sizes[-1], layer_sizes[-2]))

    return a, z, grad, nn_params


def logistic_regression(x, y, alpha=0.05, s=0, poly=0):
    """
    Runs logistic regression on the training data outputting weights as well as the cost

    :param x: training data (ndarray)
    :param y: labels for training data (ndarray)
    :param alpha: learning rate (num)
    :param s: regularization constant (num)
    :param poly: creates new features (num)
    :return: weights and cost (tuple of an ndarray and a num)
    """

    # create new features based off of powers of current features
    # allows the model to fit more non-linear data
    if poly != 0:
        poly_feat = np.zeros((x.shape[0], (poly * 10)))
        for n in range(2, poly + 2):
            index = range(((n-2) * 10), ((n-2) * 10 + 10))
            poly_feat[:, index] = np.power(x[poly - 2], n)
        np.append(x, poly_feat)

    np.append(x, np.ones(x.shape[0]))

    # initialize variables
    # m := num of training examples
    # J := cost
    # count := used to limit runtime
    # theta := weights
    m = y.shape[0]
    theta = np.random.rand((x.shape[1]), 1)
    count = 0
    J = 1

    def cost_function(x, y, theta, m, s=0):
        """
        :return: num
        """
        # hypothesis
        h = sigmoid(np.dot(x, theta))

        # calculate cost for each example
        temp = (1/m) * ((np.dot(np.transpose(np.negative(y)), np.log(h))) - np.dot((1-np.transpose(y)), np.log(1-h)))
        # total cost
        temp = temp.sum((0, 1))
        # regularize
        temp += (s / (2 * m)) * np.sum(theta[:-1])
        return temp

    def grad(x, y, theta, m, alpha, s=0):
        """
        Calculates the partial derivatives of the cost function
        :return: ndarry
        """
        # hypothesis
        h = sigmoid(np.dot(x, theta))

        theta -= (alpha / m) * (np.dot(np.transpose(x), (h - y)))
        # regularization
        theta[:-1] -= ((s / m) * theta[:-1])
        return theta

    # run gradient descent until cost drops below a given threshold or the function times out
    while J > 0.001 and count < 100000:
        # update theta
        theta = grad(x, y, theta, m, alpha, s)
        # calculate cost every 1000 iterations
        if count % 1000 == 0:
            J = cost_function(x, y, theta, m, s)
            #print(J)

        count += 1

    return theta, cost_function(x, y, theta, m, s)


def neural_net(hiden_layer_sizes, x, y, s=0, g=0, alpha=0.05):
    """

    :param hiden_layer_sizes: tuple of ints
    :param numLabels: int
    :param num_hiden_layers: int
    :param x: array of data
    :param y: vector
    :param s: int: lambda
    :return: tuple (cost, gradients)
    """
    # init variables
    layer_sizes = [x.shape[0]]
    layer_sizes.extend(hiden_layer_sizes)
    layer_sizes.append(y.shape[0])
    num_hiden_layers = len(hiden_layer_sizes)
    m = x.shape[1]
    a, z, grad, nn_params = init_nn(layer_sizes, num_hiden_layers, x)

    # prints shapes of various objects for debugging purposes
    # debug_shapes(a, z, nn_params, grad)

    # run gradient descent
    count = 0
    while count <= 100000:
        a, z = feedforward(a, z, nn_params)
        nn_params = backprop(a, z, nn_params, grad, x, y, m, alpha, s)
        if count % 5000 == 0:
            if count == 0:
                print("Iteration: " + str(count//5000))
            else:
                print(str(count//5000))

        count += 1

    accuracy = test_nn(hiden_layer_sizes, x, y, nn_params)
    return nn_params, accuracy


def sigmoid(x, L=1, k=1,x0=0):
    """
    Preforms sigmoid function element wise on x

    L, k, x0 are constants
    :param x: ndarray
    :param L: num
    :param k: num
    :param x0: num
    :return: ndarray
    """
    return L/(1 + (np.exp((np.negative(k*(x - x0))))))


def sigmoid_derivative(x, L=1, K=1, x0=0):
    """

    :param x: ndarray
    :return: ndarray
    """
    s = sigmoid(x, L, K, x0)
    return s * (1 - s)


def test(x_test, y, theta):
    """
    predicts y based on theta then calculates percent correct
    :param x_test: ndarray
    :param y: ndarray
    :param theta: ndarray
    :return: float
    """
    h = sigmoid(np.dot(feature_norm(x_test), theta))
    for n in h:
        if n <= 0.5:
            n = 0
        else:
            n = 1

    score = np.abs((y - h))
    return 100 - (np.mean(score) * 100)


def test_nn(hiden_layer_sizes, x_test, y, nn_params):
    def init_test_nn(layer_sizes, num_hiden_layers, x):
        a = {}
        z = {}
        a[0] = x_test
        for n in range(1, (num_hiden_layers + 2)):
            size = (layer_sizes[n], x_test.shape[1])
            a[n], z[n] = np.ones(size), np.ones(size)

        return a, z

    # init variables
    layer_sizes = [x_test.shape[0]]
    layer_sizes.extend(hiden_layer_sizes)
    layer_sizes.append(y.shape[0])
    numLayers = len(layer_sizes)
    num_hiden_layers = len(hiden_layer_sizes)
    a, z = init_test_nn(layer_sizes, num_hiden_layers, x_test)

    # calculate hypothesis
    a, z = feedforward(a, z, nn_params)
    h = np.squeeze(np.round(np.transpose(a[numLayers-1])))
    y = np.squeeze(y)

    score = np.abs((y - h))
    return (np.mean(score) * 100)