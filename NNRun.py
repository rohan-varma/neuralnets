import numpy as np
from NeuralNetwork import NeuralNetwork
from load_mnist import MNIST_Loader

if __name__ == '__main__':
    a = MNIST_Loader()
    X_train, y_train = a.load_mnist('../data')
    X_test, y_test = a.load_mnist('../data', 't10k')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    nn = NeuralNetwork(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)
    nn.fit(X_train, y_train, print_progress=True)
