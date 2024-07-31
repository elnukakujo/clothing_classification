from display import display_image
from image_preprocess import load_data

import numpy as np

x_train, y_train, x_test, y_test = load_data()

# Initialize parameters
n_x=x_train.shape[0]
m=x_train.shape[1]
W=np.zeros((n_x, m))
b=np.zeros((1, m))

# Define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Define the forward propagation
print(W.shape, x_train.shape, b.shape)
z=np.dot(W.T,x_train)+b
a=sigmoid(z)
print(a)


# If we want to print some images from the dataset
print_image = False
if print_image:
    index = 0
    display_image(index, x_train, y_train)