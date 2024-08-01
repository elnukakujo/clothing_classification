from display import display_image, display_metrics
from image_preprocess import load_data
from gradient_descent import gradient_descent

import numpy as np

x_train, y_train, x_test, y_test, x_test_orig = load_data()

model = gradient_descent()
parameters, costs, train_accs, test_accs = model.training(x_train, y_train, x_test, y_test, 2000)

#Prediction
y_hat=model.prediction(x_test) #, x_test_orig, 1000

#Display test accuracy
print(f"The test accuracy is {np.round(model.compute_accuracy(x_test, y_test)*100)}%")

#If we want to display an image
print_image = False
if print_image:
    index = 0
    display_image(x_train[index], y_train[index])
    
print_cost = True
if print_cost:
    display_metrics(costs, train_accs, test_accs)