"""Several functions for performing cross-validation and hyperparameter tuning."""
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

def split_data(X, y, k = 10):
    """Splits data into k portions for k-fold CV."""
    X_split = np.array_split(X, k)
    y_split = np.array_split(y, k)
    return X_split, y_split


def cross_validate(classifier, X, y, k = 10):
    """Performs cross validation to return average training and testing error
    Params:
        classifier: a classifier with a fit(X, y) and predict(y) API
        X: dataset of training examples
        y: dataset of labels for X
        k: number of portions to split data, default is 10.
    Returns:
        mean_train_error: the mean training error across the k splits
        mean_test_error: the mean testing error across the k splits
    """
    # split the data
    X_split, y_split = split_data(X, y, k)
    # for every k, train & evaluate a classifier
    training_errors, testing_errors = [], []
    for i in range(k):
        # train on D - D(k), test on D(k)
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate([X_split[j] for j in range(len(X_split))
                                  if j!=i])
        y_train = np.concatenate([y_split[j] for j in range(len(y_split))
                                  if j!=i])
        # train and test the model for our particular split of data. This split
        # will vary depending on k, so our training and validation sets
        # are different each time.
        train_error, test_error = get_errors_already_split(classifier, X_train,
                                                           y_train, X_test,
                                                           y_test,
                                                           num_iterations=1)
        training_errors.append(train_error)
        testing_errors.append(test_error)
    # average the errors across the k trials and return.
    mean_train_error = np.mean(np.array(training_errors), axis=0)
    mean_test_error = np.mean(np.array(testing_errors), axis=0)
    return mean_train_error, mean_test_error


def get_errors_already_split(classifier, X_train, y_train, X_test, y_test,
                             num_iterations=100):
    """Returns the average training and test error over a specified number of
    iterations.
    Params:
        classifier: a classifier with a fit(X, y) and predict(y) API
        X_train: training dataset of examples
        y_train: training labels
        X_test: testing dataset of examples
        y_test: testing dataset of labels
    Returns:
        train_error, test_error: the average training and testing errors of the
        classifier.
    """
    train_error, test_error = 0.0, 0.0
    for i in range(num_iterations):
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        # compute training and testing error
        train_error+=1- metrics.accuracy_score(y_true=y_train,
                                               y_pred=y_train_pred,
                                               normalize=True)
        test_error+=1 - metrics.accuracy_score(y_true=y_test,
                                               y_pred=y_test_pred,
                                               normalize=True)
    train_error /=num_iterations
    test_error /=num_iterations
    return train_error, test_error


def get_train_test_error(classifier, X, y, num_iterations = 100, split = 0.2):
    """Returns the average training and test error over a specified number of
    iterations, for a specified split of the data.
    Params:
        classifier: a classifier with a fit(X, y) and predict(y) API
        X: the training dataset of examples
        y: the testing dataset of examples
        num_iterations: number of iterations to run fit() and predict()
        split: the propoprtion of data that should be reserved for validation.
    """

    train_error, test_error = 0.0, 0.0
    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split,
                                                            random_state=i)
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_error+=1- metrics.accuracy_score(y_true=y_train,
                                               y_pred=y_train_pred,
                                               normalize=True)
        test_error+=1 - metrics.accuracy_score(y_true=y_test,
                                               y_pred=y_test_pred,
                                               normalize=True)
    train_error /=num_iterations
    test_error /=num_iterations
    return train_error, test_error

def get_best_hyperparameters(X, y, k =10, hidden_layer_neurons = [30],
                             l2_reg_strengths = [0.1], learning_rates = [0.001],
                             momentums = [0.00001], decay_rates =[0.00001],
                             dropouts = [True, False], minibatch_sizes = [500]):
    """Hyperparameter tuning with grid-search and k-fold CV. This algorithm searches
    the entire specified Hyperparameter space for the best combination of hyperparameters,
    testing each set of hyperparameters with k-fold cross validation
    Params:
    X: training dataset
    y: testing dataset
    k: number of partititions for k-fold CV
    lists: the hyperparameters from which to get the best combination.
    """
    testing_errors = []
    X_split, y_split = split_data(X, y, k)
    for num_neurons in hidden_layer_neurons:
        for l2_reg_strength in l2_reg_strengths:
            for learning_rate in learning_rates:
                for momentum in momentums:
                    for decay_rate in decay_rates:
                        for do_dropout in dropouts:
                            for minibatch_size in minibatch_sizes:
                                cur_test_error = 0.0
                                for i in range(k):
                                    X_test, y_test = X_split[i], y_split[i]
                                    X_train = np.concatenate([X_split[j] for j in range(len(X_split)) if j!=i])
                                    y_train = np.concatenate([y_split[j] for j in range(len(y_split)) if j!=i])
                                    nn = NeuralNetwork(n_output=10, n_features=X_train.shape[1],
                                                       n_hidden =  num_neurons, l2 = l2_reg_strength,
                                                       epochs = 500, learning_rate=learning_rate,
                                                       momentum_const=momentum, decay_rate=decay_rate,
                                                       dropout=do_dropout, minibatch_size=minibatch_size)
                                    print "evaluating nn with hyperparameters " +
                                    nn.fit(X_train, y_train, print_progress=False)
                                    y_test_pred = nn.predict(X_test)
                                    cur_test_error+= 1 - metrics.accuracy_score(y_true=y_test,
                                                                                y_pred=y_test_pred,
                                                                                normalize=True)
                                cur_test_error /= k
                                testing_errors.append((num_neurons, l2_reg_strength, learning_rate,
                                                       momentum, decay_rate, do_dropout,
                                                       minibatch_size, cur_test_error))

    return max(testing_errors, key = lambda x: x[len(x)-1])


def get_best_depth(X, y, k = 10, depths = []):
    """Hyperparameter tuning with grid search and k-fold CV. Finds the optimal
    maximum depth for our classifier.
    Params:
        X: training dataset
        y: labels for training examples
        k: number of portions the data should be split into.
         If k = X.shape[0] -1, this is leave-one-out CV.
        depths: a list of depths to consider
    Returns:
        tuple of depth, test error indicating the depth corresponding to the
        lowest testing error.
    """
    if len(depths) == 0:
        depths.append(None)
    depth_to_err = {}
    depth_to_train_err = {}
    for depth in depths:
        test_errors, train_errors = [], []
        X_split, y_split = split_data(X, y, k)
        for i in range(k):
            X_test, y_test = X_split[i], y_split[i]
            X_train = np.concatenate([X_split[j] for j in range(len(X_split))
                                      if j!=i])
            y_train = np.concatenate([y_split[j] for j in range(len(y_split))
                                      if j!=i])
            dclf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
            dclf.fit(X_train, y_train)
            y_test_predictions = dclf.predict(X=X_test)
            y_train_predictions = dclf.predict(X=X_train)
            test_error = 1 - metrics.accuracy_score(y_true=y_test,
                                                   y_pred=y_test_predictions,
                                                   normalize=True)
            train_error = 1 - metrics.accuracy_score(y_true=y_train,
                                                     y_pred=y_train_predictions)
            test_errors.append(test_error)
            train_errors.append(train_error)
        # average the k performance metrics
        averaged_err = np.mean(test_errors)
        depth_to_train_err[depth] = np.mean(train_errors)
        depth_to_err[depth] = averaged_err
    print depth_to_err
    print depth_to_train_err
    plt.plot(depth_to_train_err.keys(), depth_to_train_err.values())
    plt.figure()
    plt.plot(depth_to_err.keys(), depth_to_err.values())
    plt.show()
    return min(depth_to_err.items(), key = lambda x: x[1])


if __name__ == "__main__":
    print "running tests with decision tree"
    # test it with decision tree
    print "creating dataset"
    X, y = sklearn.datasets.make_classification(n_samples = 1000, n_features=10,
                                              n_redundant=6,
                                              n_informative=4,
                                              random_state=1,
                                              n_clusters_per_class=2,
                                              n_classes=7,)

    X, y = np.array(X), np.array(y)
    d_tree = DecisionTreeClassifier(criterion="entropy")
    print "training & evaluating decision tree"
    train_err, test_err = get_train_test_error(d_tree, X, y, split=0.7)
    print "training error: " + str(train_err)
    print "testing error: " + str(test_err)
    print "getting cross validation errors"
    train_error_cv, test_error_cv = cross_validate(classifier=d_tree, X=X, y=y,
                                                   k = 10)
    print "training CV error: " + str(train_error_cv)
    print "testing cv error: " + str(test_error_cv)

    print "trying to find best depth...."
    depths = np.arange(1,15)
    best_depth, best_test_err = get_best_depth(X=X, y=y, depths = depths)
    print "best depth found: " + str(best_depth)
    print "testing error for that: " + str(best_test_err)
