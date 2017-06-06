# neuralnets
Implementations and experiments with Neural Networks. 
All of this code will be ported to Python 3 shortly.
The main file of importance is neuralnetwork.py. It contains a from-scratch implementation of a neural network with a single hidden layer, without the use of any external libraries other than numpy. 

The network achieves about 96.5% accuracy on MNIST if the hyperparameters are tuned. 

The implementation has a few of the bells and whistles that help neural networks learn better:

  - An option for using stochastic gradient descent with minibatch learning
  - Decaying the learning rate during training
  - Implementation of the momentum method for better convergence & less oscillations
  - Nesterov momentum 
  - L2 regularization to promote smaller weights
  - The dropout method that discards hidden layer activations to prevent overfitting. 
  
Another interesting file is utils/utils.py. 
It contains several of my machine learning utilities for one-hot encoding, k-fold cross-validation, splitting datasets, and hyperparameter tuning. 
