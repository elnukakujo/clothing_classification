import numpy as np

def sigmoid_cost(A, Y):
    m=Y.shape[0]
    cost=(-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return cost

def softmax_cost(A,Y):
    m=Y.shape[1]
    logprobs = np.log(A + 1e-8)
    cost = -np.sum(Y * logprobs) / m
    return cost