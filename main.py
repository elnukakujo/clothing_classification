from display import display_image, display_metrics
from image_preprocess import load_data
from models import binary_classification, multi_classification

import numpy as np

def binary(path=False,print_image=False, print_metric=False):
    x_train, y_train, x_dev, y_dev, x_dev_orig = load_data(labels=[0,1])
    
    model = binary_classification()
    if not path:
        parameters, costs, train_accs, dev_accs = model.training(x_train, y_train, x_dev, y_dev, 2000, learning_rate=0.009)
    else:
        parameters = model.load_weights('path')
    
    #Prediction
    y_hat=model.prediction(x_dev, x_dev_orig, index=1001) #, x_dev_orig, 1000
    
    #Display dev accuracy
    print(f"The dev accuracy is {np.round(model.compute_accuracy(x_dev, y_dev)*100)}%")
    
    if print_image:
        index = 0
        display_image(x_train[index], y_train[index])
        
    if print_metric:
        display_metrics(costs, train_accs, dev_accs)

def multi_class(path=False,print_image=False, print_metric=False):
    x_train, y_train, x_dev, y_dev, x_dev_orig = load_data()
    
    model=multi_classification()
    if not path:
        parameters, costs, train_accs, dev_accs = model.training(x_train,y_train, x_dev, y_dev, epochs=10, 
                                                                 learning_rate=0.008, batch_size=128,optimizer=False)
    else:
        parameters, costs, train_accs, dev_accs = model.load_weights(path)
    print(f"The dev accuracy is {np.round(model.compute_accuracy(x_dev, y_dev)*100)}%")
    
    if print_image:
        index = 0
        display_image(x_train[index], y_train[index])
    try:
        if print_metric:
            display_metrics(costs, train_accs, dev_accs)
    except UnboundLocalError or NameError:
        print("Costs, train accuracies and/or dev accuracies not defined")
        
multi_class(print_metric=True)#path="epoch_10_train_acc_0.90215_cost_1.2711365475223266",