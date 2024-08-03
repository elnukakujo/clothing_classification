# clothing_binary_classification
A 1-2day binary classification challenge using Logistic Regression and Gradient Descent only with NumPy

The dataset is from the fashion MNIST dataset containing 10 classes along more than 60000 images:

https://github.com/zalandoresearch/fashion-mnist

For the bnary classification, we use only the class tshirt 0 and trouser 1 with each 6000 training images and 1000 testing images.

## Result

### Binary Classification using logistic regression
With logistic regression, the highest test accuracy is 98%, and the lowest cost is 0.075.
![image](https://github.com/user-attachments/assets/b15b9eb0-0743-44fc-ba94-4b58f802edc0)

### Multi-class classification
For now, using a 2 layer model with tanh and softmax activation function, and cross entropy, highest train_acc=0.61885, lowest cost:1.79
![image](https://github.com/user-attachments/assets/20bacf50-150f-44f6-b12d-cdbde20129d9)
