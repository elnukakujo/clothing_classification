from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def reduce_and_flatten(x):
    x = x.reshape(x.shape[0],-1).T
    x = x / 255.0
    return x

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    filter = [np.isin(y_train, [0,1]), np.isin(y_test, [0,1])]

    x_train_filtered, y_train_filtered = x_train[filter[0]], y_train[filter[0]]
    x_test_filtered, y_test_filtered = x_test[filter[1]], y_test[filter[1]]
    
    x_train_flatten = reduce_and_flatten(x_train_filtered)
    x_test_flatten = reduce_and_flatten(x_test_filtered)
    
    if not np.array_equal(np.unique(y_train_filtered), [0, 1]):
        y_train_filtered = np.where(y_train_filtered != 0, 1, y_train_filtered)
        y_test_filtered = np.where(y_test_filtered != 0, 1, y_test_filtered)
    
    return x_train_flatten, y_train_filtered, x_test_flatten, y_test_filtered, x_test_filtered