import numpy as np
from display import display_image

def sigmoid(z):
    return 1/(1+np.exp(-z))

class gradient_descent:
    parameters=dict()
    def initialize_parameters(self, X):
        n_x=X.shape[0]
        W=np.zeros((n_x, 1)) #np.random.randn(n_x, 1)*0.01
        b=0.0
        self.parameters = {
            "W":W,
            "b":b
        }

    def forward_propagation(self,X):
        W=self.parameters["W"]
        b=self.parameters["b"]
        Z=np.dot(W.T,X)+b
        A=sigmoid(Z)
        return A

    def compute_cost(self, A, Y):
        m=Y.shape[0]
        cost=(-1/m)*np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
        return cost

    def back_propagation(self, A, X, Y, learning_rate):
        m=X.shape[0]
        dZ=A-Y
        dW=np.dot(X,dZ.T)/m
        db=np.sum(dZ)/m
        
        W=self.parameters["W"]
        b=self.parameters["b"]
        
        W-=dW*learning_rate
        b-=db*learning_rate
        
        self.parameters = {
            "W":W,
            "b":b
        }

    def prediction(self, X, X_orig=[], index=0):
        Y_prediction = np.round(self.forward_propagation(X)).astype(int)
        if len(X_orig)>0:
            display_image(X_orig[index], Y_prediction.T[index])
        return Y_prediction
            
    def compute_accuracy(self, X, Y):
        Y_prediction=self.prediction(X)
        return np.mean(Y_prediction==Y)
        

    def training(self, X, Y, x_test, y_test, steps, learning_rate=0.0075):
        # Initialize parameters
        self.initialize_parameters(X)
        costs=[]
        train_accs=[]
        test_accs=[]
        #Iterate for every step
        for i in range(0,steps-1):
            # Define the forward propagation
            A=self.forward_propagation(X)
            cost = self.compute_cost(A,Y)
            train_acc=self.compute_accuracy(X, Y)
            test_acc=self.compute_accuracy(x_test,y_test)
            #Start back propagation
            self.back_propagation(A,X,Y, learning_rate)
            if i%100==0:
                #Compute the cost and the training accuracy
                print(f"Iteration {i}: cost = {cost} ; train_acc = {np.round(train_acc*100)}% ; test_acc = {np.round(test_acc*100)}%")
            costs.append(cost)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        return self.parameters, costs, train_accs, test_accs