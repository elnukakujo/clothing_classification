import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z/np.sum(e_z, axis=0, keepdims=True)

def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1-np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return np.where(z >= 0, 1, 0)