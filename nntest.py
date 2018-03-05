import unittest
import numpy as np
from NeuralNetwork import NeuralNetwork

NUM_OUTPUT_LABELS = 10
NUM_FEATURES = 20
NUM_HIDDEN_UNITS = 30

class NNTest(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(NUM_OUTPUT_LABELS, NUM_FEATURES)

    def test_initialize_weights(self):
        w1, w2 = self.nn.initialize_weights()
        #rows of w1 should = NUM_HIDDEN_UNITS, cols = NUM_FEATURES + 1 (bias)
        self.assertEqual(w1.shape[0], NUM_HIDDEN_UNITS)
        self.assertEqual(w1.shape[1], NUM_FEATURES +1)
        #rows of w2 = NUM_OUTPUT_LABELS, cols = NUM_HIDDEN_UNITS + 1 (bias)
        self.assertEqual(w2.shape[0], NUM_OUTPUT_LABELS)
        self.assertEqual(w2.shape[1], NUM_HIDDEN_UNITS +1)

    def test_one_hot_encoding(self):
        #create a labels array of size num_samples, where y[i] is the corresponding label of that instance.
        y = [1, 5, 3, 0, 2, 5, 9, 4] #ie, training instance 2 has label y[i=2] = 3
        #max(y) <= NUM_OUTPUT_LABELS -1
        encoded = self.nn.encode_labels(np.asarray(y), NUM_OUTPUT_LABELS)
        for i in range(encoded.shape[1]):
            #get encoded row vector
            row = encoded[:,i]
            #get index where this vector is one, which should be the label
            label = np.where(row==1)[0]
            self.assertEqual(y[i], label)

    def test_add_bias_unit(self):
        #set up testing arrays
        X = np.zeros((5, 4))
        X_new = self.nn.add_bias_unit(X)
        arr_ones = np.ones(X_new.shape[1])
        arr_zeros = np.zeros(X_new.shape[1])
        #add bias unit in first column
        #first column should now be all ones, rest should be all zeros
        self.assertEqual(X_new[:,0].tolist(), arr_ones.tolist())
        for i in range(1, X.shape[1]):
            self.assertEqual(X_new[:, i].tolist(), arr_zeros.tolist())

        X_new_row = self.nn.add_bias_unit(X, column=False)
        self.assertEqual(X_new_row[0, :].tolist(), np.ones(X_new_row.shape[1]).tolist())





if __name__ == '__main__':
    unittest.main()
