from display import display_image, display_metrics
from image_preprocess import load_data
from models import binary_classification, multi_classification

import numpy as np

x_train, y_train, x_test, y_test, x_test_orig = load_data()#labels=[0,1]

"""model = binary_classification()
parameters, costs, train_accs, test_accs = model.training(x_train, y_train, x_test, y_test, 2000, learning_rate=0.009)"""

model=multi_classification()
parameters, costs, train_accs, test_accs = model.training(x_train,y_train, x_test, y_test, steps=200, learning_rate=0.009)

#parameters = model.load_weights('bin_cla_train_acc_0.9790833333333333.npy')

"""#Prediction
y_hat=model.prediction(x_test, x_test_orig, index=1001) #, x_test_orig, 1000

#Display test accuracy
print(f"The test accuracy is {np.round(model.compute_accuracy(x_test, y_test)*100)}%")

#If we want to display an image
print_image = False
if print_image:
    index = 0
    display_image(x_train[index], y_train[index])"""
    
print_cost = True
if print_cost:
    display_metrics(costs, train_accs, test_accs)