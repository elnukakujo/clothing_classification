from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def reduce_and_flatten(x):
    x = x.reshape(x.shape[0],-1).T
    x = x / 255.0
    return x

def one_hot_encoded(y):
    y_encoded = np.zeros((y.shape[0],np.unique(y).shape[0]))
    y_encoded[np.arange(y.shape[0]), y] = 1
    return y_encoded.T

def load_data(labels=[]):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if len(labels)!=0:
        filter = [np.isin(y_train, labels), np.isin(y_test, labels)]

        #Filters to get only the classes we want
        x_train, y_train = x_train[filter[0]], y_train[filter[0]]
        x_test, y_test = x_test[filter[1]], y_test[filter[1]]
        
        # Originally the array is filled with numbers from 0-9, we want only 0-1
        if not np.array_equal(np.unique(y_train), [0, 1]):
            y_train = np.where(y_train != 0, 1, y_train)
            y_test = np.where(y_test != 0, 1, y_test)
    else:
        #The labels needs to be onehot encoded for the softmax activation fct
        y_train = one_hot_encoded(y_train)
        y_test = one_hot_encoded(y_test)
        
    x_train_flatten = reduce_and_flatten(x_train)
    x_test_flatten = reduce_and_flatten(x_test)
    
    return x_train_flatten, y_train, x_test_flatten, y_test, x_test