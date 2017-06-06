import numpy as np
from scipy.special import expit
import sys
import os

class NeuralNetwork(object):
    """ Feedforward neural network with a single hidden layer
        Params:
        n_output: int: number of output units, equal to num class labels
        n_features: int: number of features in the input dataset
        n_hidden: int: (default 30): num hidden units
        l2: float(default: 0.0) - lambda value for L2 regularization
        epochs: int (default = 500) - passes over training set
        learning_rate: float (default: 0.001) - learning reate
        momentum_const: float (default: 0.0) - momentum constant - multiplied with gradient of previous pass through set
        decay_rate: float (default 0.0) - shrinks learning rate after each epoch
        minibatch_size: int (default: 1) - divides training data into batches for efficiency
    """

    def __init__(self, n_output, n_features, n_hidden=30, l2=0.0, epochs=500,
                 learning_rate=0.001, momentum_const=0.0, decay_rate=0.0,
                 dropout=False, minibatch_size=1,
                 optimizer = 'Gradient Descent', activation = 'relu',
                 nesterov = False, check_gradients = False, early_stop = None, metrics = ['Accuracy']):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self.initialize_weights()
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_const = momentum_const
        self.decay_rate = decay_rate
        self.dropout = dropout
        self.minibatch_size = minibatch_size
        self.nesterov = nesterov
        self.check_gradients = check_gradients
        supported_optimizers = ['Gradient Descent', 'Momentum', 'Nesterov', 'Adam', 'Adagrad', 'Adadelta', 'RMSProp']
        if optimizer not in supported_optimizers:
            print("Error: unsupported optimizer requested.")
            print("Available optimizers: {}".format(supported_optimizers))
            exit()
        else:
            self.optimizer = optimizer
        supported_activations = ['relu', 'tanh', 'sigmoid', 'maxout', 'elu']
        if activation not in supported_activations:
            print("Error: unsupported activation requested.")
            print("Available activations: {}".format(supported_activations))
        else:
            self.activation = activation
        self.early_stop = early_stop
        SUPPORTED_METRICS = ['Accuracy', 'Precision', 'Recall', 'AUC']
        for elem in metrics:
            assert elem in SUPPORTED_METRICS
        self.metrics = metrics


    def initialize_weights(self):
        """ init weights with random nums uniformly with small values
        """
        w1  = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features+1)).reshape(self.n_hidden, self.n_features + 1)
        w2  = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden+1)).reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def encode_labels(self, y, num_labels):
        """ Encode labels into a one-hot representation
            Params:
            y: array of num_samples, contains the target class labels for each training example.
            For example, y = [2, 1, 3, 3] -> 4 training samples, and the ith sample has label y[i]
            k: number of output labels
            returns: onehot, a matrix of labels by samples. For each column, the ith index will be
            "hot", or 1, to represent that index being the label.
        """
        onehot = np.zeros((num_labels, y.shape[0]))
        for i in range(y.shape[0]):
            onehot[y[i], i] = 1.0
        return onehot

    def softmax(self, v):
        """Calculates the softmax function that outputs a vector of values that sum to one.
            We take max(softmax(v)) to be the predicted label. The output of the softmax function
            is also used to calculate the cross-entropy loss
        """
        logC = -np.max(v)
        return np.exp(v + logC)/np.sum(np.exp(v + logC), axis = 0)

    def tanh(self, z, deriv=False):
        """ Compute the tanh function or its derivative.
        """
        return np.tanh(z) if not deriv else 1 - np.square(np.tanh(z))

    def relu(self, z, deriv = False):
        if not deriv:
            relud = z
            relud[relud < 0] = 0
            return relud
        deriv = z
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv


    def add_bias_unit(self, X, column=True):
        """Adds a bias unit to our inputs"""
        if column:
            bias_added = np.ones((X.shape[0], X.shape[1] + 1))
            bias_added[:, 1:] = X
        else:
            bias_added = np.ones((X.shape[0] + 1, X.shape[1]))
            bias_added[1:, :] = X

        return bias_added

    def compute_dropout(self, activations):
        """Sets half of the activations to zero
        Params: activations - numpy array
        Return: activations, which half set to zero
        """
        mult = np.random.binomial(1, 0.5, size = activations.shape)
        activations*=mult
        return activations

    def forward(self, X, w1, w2, do_dropout = True):
        """ Compute feedforward step
            Params:
            X: matrix of num_samples by num_features, input layer with samples and features
            w1: matrix of weights from input layer to hidden layer. Dimensionality of num_hidden_units by num_features + 1 (bias)
            w2: matrix of weights from hidden layer to output layer. Dimensionality of num_output_units (equal to num class labels) by num_hidden units + 1 (bias)
            dropout: If true, randomly set half of the activations to zero to prevent overfitting.
        """
        #the activation of the input layer is simply the input matrix plus bias unit, added for each sample.
        a1 = self.add_bias_unit(X)
        if self.dropout and do_dropout: a1 = self.compute_dropout(a1)
        #the input of the hidden layer is obtained by applying our weights to our inputs. We essentially take a linear combination of our inputs
        z2 = w1.dot(a1.T)
        #applies the tanh function to obtain the input mapped to a distrubution of values between 0 and 1
        a2 = self.tanh(z2)
        #add a bias unit to activation of the hidden layer.
        a2 = self.add_bias_unit(a2, column=False)
        if self.dropout and do_dropout: a2 = self.compute_dropout(a2)
        # compute input of output layer in exactly the same manner.
        z3 = w2.dot(a2)
        # the activation of our output layer is just the softmax function.
        a3 = self.softmax(z3)
        return a1, z2, a2, z3, a3

    def get_cost(self, y_enc, output, w1, w2):
        """ Compute the cost function.
            Params:
            y_enc: array of num_labels x num_samples. class labels one-hot encoded
            output: matrix of output_units x samples - activation of output layer from feedforward
            w1: weight matrix of input to hidden layer
            w2: weight matrix of hidden to output layer
            """
        cost = - np.sum(y_enc*np.log(output))
        # add the L2 regularization by taking the L2-norm of the weights and multiplying it with our constant.
        l2_term = (self.l2/2.0) * (np.sum(np.square(w1[:, 1:])) + np.sum(np.square(w2[:, 1:])))
        cost = cost + l2_term
        return cost/y_enc.shape[1]

    def backprop(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Computes the gradient using backpropagation
            Params:
            a1: array of n_samples by features+1 - activation of input layer (just input plus bias)
            a2: activation of hidden layer
            a3: activation of output layer
            z2: input of hidden layer
            y_enc: onehot encoded class labels
            w1: weight matrix of input layer to hidden layer
            w2: weight matrix of hidden to output layer
            returns: grad1, grad2: gradient of weight matrix w1, gradient of weight matrix w2
        """
        #backpropagate our error
        sigma3 = a3 - y_enc
        z2 = self.add_bias_unit(z2, column=False)
        sigma2 = w2.T.dot(sigma3) * self.tanh(z2, deriv=True)
        #get rid of the bias row
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
         # add the regularization term
        grad1[:, 1:]+= (w1[:, 1:]*self.l2) # derivative of .5*l2*w1^2
        grad2[:, 1:]+= (w2[:, 1:]*self.l2) # derivative of .5*l2*w2^2
        return grad1, grad2

    def training_acc(self, X_train, y_train):
        """Calculate the training accuracy. Requires passing through the entire dataset."""
        y_train_pred = self.predict(X_train)
        diffs = y_train_pred - y_train
        count = 0.
        for i in range(y_train.shape[0]):
            if diffs[i] != 0:
                count+=1
        return 100 - count*100/y_train.shape[0]

    def predict(self, X, dropout = False):
        """Generate a set of predicted labels for the input dataset"""
        a1, z2, a2, z3, a3 = self.forward(X, self.w1, self.w2, do_dropout = False)
        #z3 is of dimension output units x num_samples. each row is an array representing the likelihood that the sample belongs to the class label given by the index...
        #ex: first row of z3 = [0.98, 0.78, 0.36]. This means our network has 3 output units = 3 class labels. And this instance most likely belongs to the class given by the label 0.
        y_pred = np.argmax(a3, axis = 0)
        return y_pred


    def fit(self, X, y, print_progress=True):
        """ Learn weights from training data
            Params:
            X: matrix of samples x features. Input layer
            y: target class labels of the training instances (ex: y = [1, 3, 4, 4, 3])
            print_progress: True if you want to see the loss and training accuracy, but it is expensive.
        """
        X_data, y_data = X.copy(), y.copy()
        y_enc = self.encode_labels(y, self.n_output)
        # PREVIOUS GRADIENTS
        prev_grad_w1 = np.zeros(self.w1.shape)
        prev_grad_w2 = np.zeros(self.w2.shape)
        print("fitting")

        #pass through the dataset
        for i in range(self.epochs):
            previous_accuracies = []
            self.learning_rate /= (1 + self.decay_rate*i)
            mini = np.array_split(range(y_data.shape[0]), self.minibatch_size)
            for idx in mini:
                #feed feedforward
                a1, z2, a2, z3, a3= self.forward(X_data[idx], self.w1, self.w2)
                cost = self.get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)

                #compute gradient via backpropagation

                grad1, grad2 = self.backprop(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)

                if self.check_gradients:
                    # compute numerical gradient
                    h= 1e-5
                    w1_h = self.w1 + h
                    _, _, _, _, out1 = self.forward(X_data[idx], w1_h, self.w2, do_dropout = False)
                    w1_h = self.w1 - h
                    _, _, _, _, out2 = self.forward(X_data[idx], w1_h, self.w2, do_dropout = False)
                    numerical_deriv_w1 = (out1 - out2) /float(2 * h)
                    analytical = np.sum(grad1)
                    numerical = np.sum(numerical_deriv_w1)
                    w1_grad_error = np.abs(analytical - numerical) / np.max(np.abs(analytical), np.abs(numerical))
                    #print("gradient error: {}".format(w1_grad_error))


                # update parameters, multiplying by learning rate + momentum constants

                w1_update, w2_update = self.learning_rate*grad1, self.learning_rate*grad2
                if self.nesterov:
                    # v_prev = v # back this up
                    # v = mu * v - learning_rate * dx # velocity update stays the same
                    # x += -mu * v_prev + (1 + mu) * v # position update changes form
                    # psuedocode from http://cs231n.github.io/neural-networks-3/#sgd
                    v1 = self.momentum_const * prev_grad_w1 - w1_update
                    v2 = self.momentum_const * prev_grad_w2 - w2_update
                    self.w1 += -self.momentum_const * prev_grad_w1 + (1 + self.momentum_const) * v1
                    self.w2 += -self.momentum_const * prev_grad_w2 + (1 + self.momentum_const) * v2
                else:
                    # gradient update: w += -alpha * gradient.
                    # use momentum - add in previous gradient mutliplied by a momentum hyperparameter.
                    self.w1 += -(w1_update + (self.momentum_const*prev_grad_w1))
                    self.w2 += -(w2_update + (self.momentum_const*prev_grad_w2))
                prev_grad_w1, prev_grad_w2 = w1_update, w2_update

            if print_progress and (i+1) % 50==0:
                print "Epoch: " + str(i+1)
                print "Loss: " + str(cost)
                print("gradient error: {}".format(w1_grad_error))
                acc = self.training_acc(X, y)
                previous_accuracies.append(acc)
                if self.early_stop is not None and len(previous_accuracies) > 3:
                    if abs(previous_accuracies[-1] - previous_accuracies[-2]) < self.early_stop and abs(previous_accuracies[-1] - previous_accuracies[-3]) < self.early_stop:
                        print("Early stopping, accuracy has stayed roughly constant over last 100 iterations.")
                        break

                print "Training Accuracy: " + str(acc)

        return self
