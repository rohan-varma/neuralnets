import numpy as np

# Optimizations module - implements various optimization functions

def vanilla_gd(params, grad1, grad2):
    # The vanilla gradient descent algorithm
    # w_new = w_old - lr * grad
    if len(params) == 0:
        raise ValueError("params must have a length of at least one, and params[0] should be the learning rate.")
    learning_rate = params[0]
    w1_update, w2_update = learning_rate * grad1, learning_rate * grad2
    return w1_update, w2_update

def momentum_optimizer(params, grad1, grad2, prev_grad_1, prev_grad_2, nesterov = False):
    # Gradient descent with momentum.
    learning_rate = params[0]
    momentum_rate = params[1]

    w1_update, w2_update = learning_rate * grad1, learning_rate * grad2

    if nesterov:
        # momentum update is the same
        v1 = momentum_rate * prev_grad_1 - w1_update
        # gradient is taken at the place where the momentum takes the param
        v2 = momentum_rate * prev_grad_2 - w2_update
        w1_update = momentum_rate * prev_grad_1 - (1 + momentum_rate) * v1
        w2_update = momentum_rate * prev_grad_2 - (1 + momentum_rate) * v2
    else:
        # the update just takes on an additional constant * the previous gradient.
        w1_update = (w1_update + momentum_rate * prev_grad_1)
        w2_update = (w2_update + momentum_rate * prev_grad_2)

    return w1_update, w2_update
