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

def momentum_optimizer(params, grad1, grad2, prev_grad_1, prev_grad_2):
    # Gradient descent with momentum.
    pass
