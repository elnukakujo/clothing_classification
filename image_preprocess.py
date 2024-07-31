from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def reduce_and_flatten(x):
    x = x / 255.0
    x = x.reshape((x.shape[1]*x.shape[2],x.shape[0]))
    return x

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    filter = [np.isin(y_train, [0,1]), np.isin(y_test, [0,1])]

    x_train, y_train = x_train[filter[0]], y_train[filter[0]]
    x_test, y_test = x_test[filter[1]], y_test[filter[1]]
    
    x_train = reduce_and_flatten(x_train)
    x_test = reduce_and_flatten(x_test)
    
    return x_train, y_train, x_test, y_test