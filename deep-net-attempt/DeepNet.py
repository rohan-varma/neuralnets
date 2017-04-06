import numpy as np
from scipy.special import expit
import sys
import os

# ________________________Layer Class______________________________


class HiddenLayer(object):
    def __init__(self, prev_layer_neurons, this_layer_neurons=30):
        matrix_size = this_layer_neurons*(prev_layer_neurons+1)
        self.weights = np.random.uniform(-1.0, 1.0, size=matrix_size).reshape
        (this_layer_neurons, prev_layer_neurons+1)
        self.prev_neurons = prev_layer_neurons
        self.layer_neurons = this_layer_neurons

# ________________________Deep Neural Net______________________________


class DeepNeuralNet(object):
    """ Feedforward neural network with a single hidden layer
        Params:
        n_output: int: number of output units, equal to num class labels
        n_features: int: number of features in the input dataset
        n_hidden: int: (default 30): num hidden units
        l1: float (default: 0.0) - lambda value for L1 regularization
        l2: float(default: 0.0) - lambda value for L2 regularization
        epochs: int (default = 500) - passes over training set
        eta: float (default: 0.001) - learning reate
        alpha: float (default: 0.0) - momentum constant - multiplied with
        gradient of previous pass through set
        decrease_const: float (default 0.0) - shrinks learning rate after each
        epoch: eta = eta / (1 + epoch*decrease_const)
        shuffle: bool (default: True) - shuffles training data each pass to
        prevent circles
        minibatches: int (default: 1) - divides training data into batches
        for efficiency
        random_state: int (default: None) - sets random state for initializing
        weights
    """
    def __init__(self, output_units, input_features, layers=[], l1=0.0, l2=0.0,
                 num_epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
                 shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.output_units = output_units
        self.input_features = input_features
        self.hidden_layers = layers
        if len(self.hidden_layers) == 0:
            self.initialize_single_layer()
        self.output_weights = self.initialize_output_weights()
        self.l1, self.l2 = l1, l2
        self.num_epochs = num_epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def initialize_single_layer(self):
        l = HiddenLayer(prev_layer_neurons=self.input_features)
        self.hidden_layers.append(l)

    def initialize_output_weights(self):
        last_layer_neurons = self.hidden_layers[-1].layer_neurons
        w_out = np.random.uniform(-1.0, 1.0, size=self.output_units*(last_layer_neurons+1)).reshape(self.output_units, last_layer_neurons + 1)
        return w_out

# ______________________________ Auxilary Methods__________________________

    def one_hot_encode(self, labels, total_num_labels):
        """ one hot encode: create a num_labels x labels matrix with the labels
            one hot encoded
        """
        one_hot_matrix = np.zeros((total_num_labels, labels.shape[0]))
        for index, value in enumerate(labels):
            one_hot_matrix[value][index] = 1.0

        return one_hot_matrix

    def sigmoid(self, z, derivative=False):
        sig = expit(z)
        return sig if not derivative else sig * (1.0 - sig)

    def add_bias_unit(self, matrix, rowWise=True):
        if rowWise:
            new_matrix = np.ones((matrix.shape[0] + 1, matrix.shape[1]))
            new_matrix[1:, :] = matrix
            return new_matrix
        else:
            new_matrix = np.ones((matrix.shape[0], matrix.shape[1] + 1))
            new_matrix[:, 1:] = matrix
            return new_matrix


# _________________ L1 & L2 Regularization Methods ________________________

    def l1_reg_cost(self, l1_reg_const, w1, w2):
        """Compute l1 regularization cost """
        return (l1_reg_const/2.0) * (np.abs(w1[:, 1:]).sum() +
                                np.abs(w2[:, 1:]).sum())

    def l2_reg_cost(self, l2_reg_const, w1, w2):
        return (l2_reg_const/2.0) * (np.sum(w1[:, 1:] ** 2) +
                                np.sum(w2[:, 1:] ** 2))


# __________________ Forward Prop implementation__________________________

    def forward_prop(self, input_matrix, layers=[], output_weights):
        """forward propagation
           Params: input_matrix: input data
           Layers: hidden layer(s) of neural network
           output weights: weights from last hidden layer to output layer
        """
        layer_activations = []
        activation_1 = self.add_bias_unit(input_matrix, rowWise=False)
        for hidden_layer in layers:
            # take the linear combination
            if(hidden_layer.weights.shape[1] == activation_1.shape[0]):
                z = hidden_layer.weights.dot(activation_1)
            else:
                z = hidden_layer.weights.dot(activation_1.T)
            activation_2 = self.sigmoid(z)
            activation_2 = self.add_bias_unit(activation_2, rowWise=True)
            layer_activations.append(activation_1)
            activation_1 = activation_2
        if output_weights.shape[1] == activation_1.shape[0]:
            z3 = output_weights.dot(activation_1)
        else:
            z3 = output_weights.dot(activation_1.T)
        final_activation = self.sigmoid(z3)
        return z3, final_activation, layer_activations

# __________________ Backpropagation implementation__________________________

    def backpropagate(layer_inputs, layer_activations, layer_weights):
        pass

# __________________ Learning & Predicting Functions______________________

    def predict(self, data):
        if len(data.shape != 2):
            raise AttributeError("data should have 2 dims")


    def fit(self, features, labels, verbose=True):
        pass
