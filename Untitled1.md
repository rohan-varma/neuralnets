
### Defining the Learning Problem 

In supervised learning problems, we're given a training dataset that contains pairs of input instances and their corresponding labels. For example, in the MNIST dataset, our input instances are images of handwritten digits, and our labels are a single digit that indicate the number written in the image. To input this training data to a computer, we need to numerically represent our data. Each image in the MNIST dataset is a 28 x 28 grayscale image, so we can represent each image as a vector $$ \bar{x} \in R^{784} $$. The elements in the vector x are known as features, and in this case they're values in between 0 and 255. Our labels are commonly denoted as y, and as mentioned, are in between 0 and 9.

We can think of this dataset as a sample from some probability distribution over the feature/label space, known as the data generating distribution. Specifically, this distribution gives us the probability of observing any particular (x, y) pairss for all (x, y) pairs in the cartesian product $$ X x Y $$. Intuitively, we would expect that the pair that consists of an image of a handwritten 2 and the label 2 to have a high probablity, while a pair that consists of a handwritten 2 and the label 9 to have a low probability.

Unfortunately, we don't know what this data generating distribution is parametrized by, and this is where machine learning comes in: we aim to learn a function h that maps feature vectors to labels as accurately as possible. This function should generalize well: we don't just want to learn a function that produces a flawless mapping on our training set. The function needs to be able to generalize over all unseen examples in the distribution. With this, we can introduce the idea of the loss function, a function that quantifies how off our prediction is from the true value. The loss function gives us a good idea about our model's performance, so over the entire population of (feature vector, label) pairs, we'd want the expectation of the loss to be low. Therefore, we want to find h(x) that minimizes the following function:

$$ E[L(y, h(\bar{x}))] = \sum_{(\bar{x}, y) \in D} p(x, y)L(y, h(x)) $$

However, there's a problem here: we can't compute p(x, y), so we have to resort to approximations of the loss function based on the training data that we do have access to. To approximate our loss, it is common to sum the loss function's output across our training data, and then divide it by the number of training examples to obtain an average loss, known as the training loss: 

$$ \frac{1}{N} \sum_{i=1}^{N} L(y_i, h(x_i)) $$

There are several different loss functions that we can use in our neural network to give us an idea of how well it is doing. The function that I ended up using was the cross-entropy loss, which will be discussed a bit later. 


In the space of neural networks, the function h(x) we will find will consist of several operations of matrix multiplications followed by applying nonlinearity functions. The basic idea is that we need to find the parameters of this function that both produce a low training loss and generalize well to unseen data. With our learning problem defined, we can get on to the implementation of the network: 

### Precursor: A single Neuron

In the special case of binary classification, we can model an artificial neuron as receiving a linear combination of our inputs, and then computing a function that returns either 0 or 1, which is the predicted label of the input. 

The weights are applied to the inputs, which are just the features of the training instance. Then, as a simple example of a function an artificial neuron can compute, we take the sign of the resulting number, and map that to a prediction. So the following is the neural model of learning: 

There's a few evident limitations to this kind of learning - for one, it can only do binary classification. Moreover, this neuron can only linearly separate data, and therefore this model assumes that the data is indeed linearly separable. Deep neural networks are capable of learning representations that model the nonlinearity inherent in many data samples. The idea, however, is that neural networks are just made up of a bunch of these neurons, which by themselves, are pretty simple, but extremely powerful when they are combined. 

### From Binary Classification to Multinomial Classfication

In the context of our MNIST problem, we're interested in producing more than a binary classification - we want to predict one label out of a possible ten. One intuitive way of doing this is simply training several classifiers - a one classifier, a two classifier, and so on. We don't want to train multiple models separately though, we'd like a single model to learn all the possible different classifications. 

If we consider our basic model of a neuron, we see that it has one vector of weights that it applies to determine a label. What if we had multiple vectors - a matrix - of weights instead? Then, each row of weights could represent a separate classifier. To see this clearly, we can start off with a simple linear mapping: 

$ a = W^{T}x + b $

For our MNIST problem, x is a vector with 784 components, and W was originally a single vector with 784 values. However, if we modified W to be a matrix instead, we get multiple rows of weights, each of which can be applied to the input x via a matrix multiplication. Since we want to be able to predict 10 different labels, we can let W be a 10 x 784 matrix, and the matrix product W^{T}x will produce a column vector of values that represent the output of 10 separate classifiers, where the weights for each classifier is given by the rows of W. 

Now that we have a vector of outputs, we'd like to figure out the most likely label. To do this, we can map our 10 dimensional vector to another 10 dimensional vector which each value is in the range (0, 1), and the sum of all values is 1. This is known as the softmax function. We can use the output of this function to represent a probability distribution: each value gives us the probability of the input x mapping to a particular label y. Here's an illustration of our model so far: 


Next, we can use our loss function discussed previously to evaluate how well our classifier is doing. Using the gradient descent algorithm, we can learn a particular matrix of weights that performs well and produces a low loss. 

Now that we've figured out how to linearly model multilabel classification, we can create a basic neural network. Consider what happens when we combine the idea of artificial neurons with our logistic classifier. Instead of computing a linear combination and immediately passing the result to a softmax function, we can have an intermediate step: pass the output of our linear combination to a vector of artificial neurons, that each compute a nonlinear function. Then, we can take a linear combination with a vector of weights for each of these outputs, and pass that into our softmax function. This "intermediate step" is actually known as a hidden layer, and we have complete control over it, meaning that among other things, we can vary the number of parameters or connections between weights and neurons to obtain an optimal network. It's also important to notice that we can stack an arbitrary amount of these hidden layers between the input and output of our network, and we can tune these layers individually. This lets us make our network as deep as we want it. Here's what our model looks like now: 


### Implementing the Neural Network

With a bit of background out of the way, we can actually begin implementing our network. If we're going to implement a neural network with one hidden layer of arbitrary size, we need to initalize two matrices of weights: one to multiply with our inputs to feed into the hidden layer, and one to multiply with the outputs of our hidden layer, to feed into the softmax layer. Here's how we can initialize our weights:

Next, its important to do a bit of preprocessing on our data so that we can represent it easily in our model. Most notably, this includes implementing one-hot encoding. Since our softmax classifier outputs a column vector of probabilities that indicate its prediction, we want an easy way to compare its output to our training labels as part of our loss function. TO do this, instead of representing each label with their number, we can represent a particular label i with a column vector of 10 dimensions, where the ith element is "hot", or 1. As an example, here's a couple one-hot encodings: 

The above now helps us implement our loss function. As discussed, we are using the average cross-entropy loss to gauge the accuracy of our classifier. The cross-entropy loss is defined as follows: 

Essentially, this sums over each index of the a particular label vector and multiplies it wit the log of our prediction at that index. Since only one index is not zero for each label vector, this computation will happen only once. We can think of the inner sum as being the log of the probability indicated by our softmax function at index i, where i is the true value of the label: 

If this probability is small, this means our classifier mispredicted the label, and the loss will be high. Similarly, if the probability is close to 1, the loss will be very small. We accumulate this loss across all training instances and average it. Here's an implementation of the loss function: 

Now with the loss function defined, we can begin to implement the fit function for our neural network, the function in which we will learn the weights that parametrize our model from our training data. The fit function consists of two essential ingredients: forward propagation and backwards propagation. In the forward propagation step, we make a prediction for each instance of our training data. This is done by propagating it through the network - applying weights to the inputs, sending that to a nonlinearity, applying weights to that, and finally getting the prediction as the output of the softmax function. Here's an implementation of forward propagation:

Now that we've got predictions for our labels, we want to adjust our weights so that our network does better next time we forward propagate. This is how a neural network learns: it makes a prediction, figures out how inaccurate it was, and updates the weights based off of that information. This is known as gradient descent: iteratively adjusting the weights by their gradient/partial derivative evaluated with respect to the loss function, until we reach (hopefully) a global minima. Computing the partial derivatives with respect to our loss function turns out to be an interesting math problem. 

The function we're trying to differentiate is given by forward propagation as above. Using the chain rule, we can compute a sequence of derivatives that get us the partial derivatives with respect to the weights: 




### Fine-tuning gradient descent
Talk about SGD, learning rate, decaying learning rate, and momentum.

### Preventing our network from overfitting
Talk about L1 and L2 regularization and dropout 


### Tuning Hyperparameters with K-Fold Cross Validation


### 



```python

```
