import numpy as np

def backprop(a, z, nn_params, grad, x, y, m, alpha, s):
    """
    Runs back propagation algorithm

    :param z: dict of ndarraies
    :param a: dict of ndarraies, g(z)
    :param nn_params: dict of ndarraies, parameters
    :param grad: dict of ndarraies, gradients
    :param x: ndarray, data
    :param y: ndarray, labels
    :param m: int, number of training examples
    :param alpha: float, learning rate
    :param s: float, regularization constant
    :return: dict of ndarraies, trained parameters
    """
    # l is the number of hidden layers + the output layer
    l = len(grad)
    dz = {}

    # calculate dz and gradients for the output layer
    dz[l] = a[l] - y
    grad[l] = (1 / m) * np.dot(dz[l], np.transpose(a[l-1]))

    # calculate dz and gradients for hidden layers
    for n in range(1, l):
        # could have done this more efficiently but layer iterates backwards through the hidden layers
        layer = l - n

        dz[layer] = np.dot(np.transpose(nn_params[layer+1]), dz[layer+1]) * sigmoid_derivative(z[layer])
        grad[layer] = (1 / m) * np.dot(dz[layer], np.transpose(a[layer-1]))

        # regularization
        grad[layer] += ((s / m) * nn_params[layer])

    # update nn_params by subtracting gradients
    for n in range(1, len(nn_params) + 1):
        nn_params[n] -= (alpha * grad[n])

    return nn_params


def dict_to_ndarray(d, t=False):
    """
    Takes dict and returns an ndarray

    :param d: dict
    :param t: bool (test)
    :return: if t=True a ndarray; else a tuple of ndarrays
    """
    # if t then there are no labels
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
    """
    Prints shapes of various ndarraies in the neural network for debugging purposes

    :param z: dict of ndarraies
    :param a: dict of ndarraies, g(z)
    :param nn_params: dict of ndarraies, parameters
    :param grad: dict of ndarraies, gradients
    :return: None
    """
    shapes = [["a: "], ["z: "], ["nn_params: "], ["grad: "]]
    l = [a, z, nn_params, grad]
    for i in range(len(l)):
        for n in l[i].values():
            shapes[i].append(n.shape)

    print(shapes)
    pass


def feature_norm(x):
    """
    Scales data so norm is 1
    :param l: ndarray
    :return: ndarray
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= x_norm
    return x


def feedforward(a, z, nn_params):
    """
    Runs feed forward algorithm
    :param z: dict of ndarraies
    :param a: dict of ndarraies, g(z)
    :param nn_params: dict of ndarraies, parameters
    :return: tuple of dict of ndarraies, updated a and z
    """
    for n in range(1, len(a)):
        # linear transformation between layer given by nn_params
        z[n] = np.dot(nn_params[n], a[n-1])

        # non-linear activation function
        # TODO: switch to a better activation function (i.e. ReLU or tanh)
        a[n] = sigmoid(z[n])
    return a, z


def format_attributes(l):
    """
    Removes strings and turns letters into numbers
    :param l: list
    :return: list
    """
    # TODO: handle my data better (fill in missing data, make use of name)

    # Takes letters in cabin number replaces them w/ nums (assumes location of cabin predicts survival)
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


def init_nn(layer_sizes, num_hidden_layers, x):
    """
    initializes a, z, grad, nn_params
    :param layer_sizes: list of ints
    :param num_hidden_layers: int
    :param x: data
    :return: tuple of dicts of ndarraies
    """
    # init dicts
    a = {}
    z = {}
    nn_params = {}
    grad = {}

    # a[0] is the input layer
    a[0] = x

    # populate a and z with ndarraies w/ sizes given by layer_sizes
    for n in range(1, (num_hidden_layers + 2)):
        size = (layer_sizes[n], x.shape[1])
        a[n], z[n] = np.ones(size), np.ones(size)

    # populate grad with ndarraies and randomly initializes nn_params
    # this is horribly messy but it works
    nn_params[1] = np.random.rand(layer_sizes[1], layer_sizes[0])
    grad[1] = np.zeros((layer_sizes[1], layer_sizes[0]))
    for n in range(2, num_hidden_layers + 2):
        if n < (num_hidden_layers + 1):
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
        poly_feat = np.zeros((x.shape[1], (poly * 10)))
        for n in range(2, poly + 2):
            index = range(((n-2) * 10), ((n-2) * 10 + 10))
            poly_feat[:, index] = np.power(x[poly - 2], n)
        np.append(x, poly_feat)

    np.append(x, np.ones(x.shape[1]))

    # initialize variables
    # m = num of training examples
    # J = cost
    # count = used to limit runtime
    # theta = weights
    m = y.shape[1]
    theta = np.random.rand(1, x.shape[0])
    count = 0
    J = 1

    def cost_function(x, y, theta, m, s=0):
        """
        :return: num
        """
        # hypothesis
        h = sigmoid(np.dot(theta, x))
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
        h = sigmoid(np.dot(theta, x))
        # update theta
        theta -= (alpha / m) * (np.dot((h - y), np.transpose(x)))
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

    error = test_lr(x, y, theta)
    return theta, error


def neural_net(hidden_layer_sizes, x, y, s=0, alpha=0.05, i=100000):
    """
    Trains a neural network with a structure specified by hidden_layer_sizes.

    :param hidden_layer_sizes: tuple of ints
    :param x: ndarry, training data
    :param y: ndarray, vector of labels
    :param s: float, regularization constant
    :param alpha: float, learning rate
    :param i: int, iterations of gradient descent to run
    :return: tuple of ndarray and float, (nn_params, error)
    """

    # init variables
    layer_sizes = [x.shape[0]]
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(y.shape[0])
    num_hidden_layers = len(hidden_layer_sizes)
    m = x.shape[1]

    # init neural network
    # sets up ndarraies with appropriate shapes
    # randomly initializes nn_params
    a, z, grad, nn_params = init_nn(layer_sizes, num_hidden_layers, x)

    # prints shapes of various objects for debugging purposes
    # debug_shapes(a, z, nn_params, grad)

    # trains nn_params
    nn_params = nn_gradient_descent(a, z, nn_params, grad, x, y, m, alpha, s, i)

    # training set error for diagnosing bias/variance
    error = test_nn(hidden_layer_sizes, x, y, nn_params)

    return nn_params, error


def nn_gradient_descent(a, z, nn_params, grad, x, y, m, alpha, s, i):
    """
    Runs gradient descent
    See neural_net for what the inputs are (could write them out put docstring would be longer than the function)

    :return: dict of ndarraies, updated nn_params
    """
    # i+1 is for aesthetic purposes in print out
    for n in range(i + 1):
        a, z = feedforward(a, z, nn_params)
        nn_params = backprop(a, z, nn_params, grad, x, y, m, alpha, s)

        # prints number of iterations completes every 5000 iterations
        if n % 10000 == 0:
            print("Iteration: " + str(n))

    return nn_params


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
    Calculates the derivative of the sigmoid function element wise

    :param x: ndarray
    :return: ndarray
    """
    s = sigmoid(x, L, K, x0)
    return s * (1 - s)


def test_lr(x_test, y, theta):
    """
    Calculates error of logistic regression

    :param x_test: ndarray
    :param y: ndarray
    :param theta: ndarray
    :return: float
    """
    # predict labels
    h = sigmoid(np.dot(theta, feature_norm(x_test)))
    h = np.round(h)

    # difference between actual labels and the predicted ones
    score = np.abs((y - h))

    # average error turned into a percentage
    return np.mean(score) * 100


def test_nn(hidden_layer_sizes, x_test, y, nn_params):
    """
    Calculates error of logistic regression

    :param hidden_layer_sizes: tuple of ints
    :param x_test: ndarray, data
    :param y: ndarray, labels
    :param nn_params: dict of ndarraies, parameters
    :return: float, error
    """
    # abridged version of init_nn with only what is necessary for feed forward
    def init_test_nn(layer_sizes, num_hidden_layers, x):
        # init dicts
        a = {}
        z = {}

        # input layer
        a[0] = x_test

        # remaining layers
        for n in range(1, (num_hidden_layers + 2)):
            size = (layer_sizes[n], x_test.shape[1])
            a[n], z[n] = np.ones(size), np.ones(size)

        return a, z

    # init variables
    layer_sizes = [x_test.shape[0]]
    layer_sizes.extend(hidden_layer_sizes)
    layer_sizes.append(y.shape[0])
    num_layers = len(layer_sizes)
    num_hidden_layers = len(hidden_layer_sizes)
    a, z = init_test_nn(layer_sizes, num_hidden_layers, x_test)

    # calculate hypothesis (squeeze is unnecessary but I like it)
    a, z = feedforward(a, z, nn_params)
    h = np.squeeze(np.round(np.transpose(a[num_layers-1])))
    y = np.squeeze(y)

    # difference between actual labels and the predicted ones
    score = np.abs((y - h))

    # average error turned into a percentage
    return np.mean(score) * 100