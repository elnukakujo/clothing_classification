from display import display_image, display_metrics
from image_preprocess import load_data
from models import binary_classification, multi_classification

import numpy as np

def binary(path=False,print_image=False, print_metric=False):
    x_train, y_train, x_test, y_test, x_test_orig = load_data(labels=[0,1])
    
    model = binary_classification()
    if not path:
        parameters, costs, train_accs, test_accs = model.training(x_train, y_train, x_test, y_test, 2000, learning_rate=0.009)
    else:
        parameters = model.load_weights('path')
    
    #Prediction
    y_hat=model.prediction(x_test, x_test_orig, index=1001) #, x_test_orig, 1000
    
    #Display test accuracy
    print(f"The test accuracy is {np.round(model.compute_accuracy(x_test, y_test)*100)}%")
    
    if print_image:
        index = 0
        display_image(x_train[index], y_train[index])
        
    if print_metric:
        display_metrics(costs, train_accs, test_accs)

def multi_class(path=False,print_image=False, print_metric=False):
    x_train, y_train, x_test, y_test, x_test_orig = load_data()
    
    model=multi_classification()
    if not path:
        parameters, costs, train_accs, test_accs = model.training(x_train,y_train, x_test, y_test, steps=10000, learning_rate=0.008)
    else:
        parameters=model.load_weights(path)

    print(f"The test accuracy is {np.round(model.compute_accuracy(x_test, y_test)*100)}%")
    
    if print_image:
        index = 0
        display_image(x_train[index], y_train[index])
        
    if print_metric:
        display_metrics(costs, train_accs, test_accs)
        
multi_class(print_metric=True)