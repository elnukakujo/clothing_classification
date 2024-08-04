# clothing_classification
A 3days classification challenge using Logistic Regression and Gradient Descent only with NumPy

The dataset is from the fashion MNIST dataset containing 10 classes along more than 60000 images:

https://github.com/zalandoresearch/fashion-mnist

For the binary classification, we use only the class tshirt 0 and trouser 1 with each 6000 training images and 1000 testing images.
For the multi class classification, we use all the 10 classes.

## Result

### Binary Classification using logistic regression
With logistic regression, the highest test accuracy is 98%, and the lowest cost is 0.075.
![image](https://github.com/user-attachments/assets/b15b9eb0-0743-44fc-ba94-4b58f802edc0)

### Multi-class classification
For now, using a 2 layer model with relu and softmax activation function, and cross entropy, highest test_acc=0.8134, lowest cost:0.5130
![step_6000_train_acc_0 8257333333333333_cost_0 5130132978716956](https://github.com/user-attachments/assets/0f2418bf-ee52-4879-a881-cf146eab43a9)
PS: I tried removing the sort of speed bump on the accuracies at the start of the training by lowering the learning rate, adding another hidden layer, changing the hidden nodes, but it would generally lower the metrics performances considerably, by, it seemed, a lower point of convergence. I would enjoy hypertuning more the hyper parameters, however because I only use NumPy, the training takes around 1h each time.
