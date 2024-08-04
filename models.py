import numpy as np
from display import display_image
from activation_functions import *
from cost_functions import *
import json

class binary_classification:
    def initialize_parameters(self, X):
        n_x=X.shape[0]
        self.parameters=np.zeros((n_x+1, 1)) #np.random.randn(n_x+1, 1)*0.01

    def forward_propagation(self,X):
        W=self.parameters[:-1,:]
        b=self.parameters[-1,:]
        Z=np.dot(W.T,X)+b
        A=sigmoid(Z)
        return A

    def back_propagation(self, A, X, Y, learning_rate):
        m=X.shape[1]
        
        dZ=A-Y
        dW=np.dot(dZ,X.T)/m
        db=np.sum(dZ)/m
        
        W=self.parameters[:-1,:]
        b=self.parameters[-1,:]
        
        W-=dW*learning_rate
        b-=db*learning_rate
        
        self.parameters = np.vstack([W, b])

    def prediction(self, X, X_orig=[], index=0):
        Y_prediction = np.round(self.forward_propagation(X)).astype(int)
        if len(X_orig)>0:
            display_image(X_orig[index], Y_prediction.T[index])
        return Y_prediction
            
    def compute_accuracy(self, X, Y):
        Y_prediction=self.prediction(X)
        return np.mean(Y_prediction==Y)
        

    def training(self, X, Y, x_test, y_test, steps, learning_rate):
        # Initialize parameters
        self.initialize_parameters(X)
        costs=[]
        train_accs=[]
        test_accs=[]
        #Iterate for every step
        for i in range(1,steps+1):
            # Define the forward propagation
            A=self.forward_propagation(X)
            cost = sigmoid_cost(A,Y)
            train_acc=self.compute_accuracy(X, Y)
            test_acc=self.compute_accuracy(x_test,y_test)
            #Start back propagation
            self.back_propagation(A,X,Y, learning_rate)
            if i%10==0:
                #Compute the cost and the training accuracy
                print(f"Iteration {i}: cost = {cost} ; train_acc = {np.round(train_acc*100)}% ; test_acc = {np.round(test_acc*100)}%")
            costs.append(cost)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        self.save_weights(i, cost, train_acc)
        return self.parameters, costs, train_accs, test_accs
    def save_weights(self, step, cost, train_acc):
        path = f'saved_models/binary/step_{step}_train_acc_{train_acc}_cost_{cost}.npy'
        np.save(path, self.parameters)
        return path
    def load_weights(self, path_name):
        self.parameters=np.load(f'saved_models/binary/{path_name}')
        return self.parameters
    
class multi_classification:
    def initialize_parameters(self,X,Y):
        n_0=X.shape[0]
        n_1=70
        n_2=Y.shape[0]
        W1=np.random.randn(n_1,n_0)*0.01
        b1=np.zeros((n_1,1))
        W2=np.random.randn(n_2,n_1)*0.01
        b2=np.zeros((n_2,1))

        self.parameters={
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
        }
        return self.parameters
    def propagate(self, X):
        W1=self.parameters["W1"]
        b1=self.parameters["b1"]
        W2=self.parameters["W2"]
        b2=self.parameters["b2"]

        Z1=np.dot(W1,X)+b1
        A1=relu(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=softmax(Z2)

        self.cache={
            "Z1":Z1,
            "A1":A1,
            "A2":A2
        }
        return A2
    def backprop(self, X, Y, learning_rate):
        m=X.shape[1]
        
        W2=self.parameters["W2"]
        Z1=self.cache["Z1"]
        A1=self.cache["A1"]
        A2=self.cache["A2"]
        
        dZ2=A2-Y
        dW2=np.dot(dZ2,A1.T)/m
        db2=np.sum(dZ2,axis=1,keepdims=True)/m
        dZ1=np.dot(W2.T,dZ2)*(drelu(Z1))
        dW1=np.dot(dZ1,X.T)/m
        db1=np.sum(dZ1, axis=1, keepdims=True)/m
        
        self.parameters["W1"] -= dW1*learning_rate
        self.parameters["b1"] -= db1*learning_rate
        self.parameters["W2"] -= dW2*learning_rate
        self.parameters["b2"] -= db2*learning_rate
        
    def prediction(self, X):
        Y_prediction = self.propagate(X)
        return Y_prediction
    
    def compute_accuracy(self, X, Y):
        A_L=self.prediction(X)
        num_examples = A_L.shape[1]
        max_indices = np.argmax(A_L, axis=0)
        Y_prediction = np.zeros_like(A_L)
        Y_prediction[max_indices, np.arange(num_examples)] = 1
        return np.mean(np.all(Y_prediction == Y, axis=0))
    
    def save_weights(self, step, train_acc, cost):
        path=f'saved_models/multi_class/step_{step}_train_acc_{train_acc}_cost_{cost}.json'
        parameters = {k: v.tolist() for k, v in self.parameters.items()}
        with open(path,'w') as json_file:
            json.dump(parameters, json_file, indent=4)
        return path
    
    def load_weights(self, path):
        with open(f'saved_models/multi_class/{path}.json','r') as json_file:
            parameters=json.load(json_file)
        self.parameters={k: np.array(v) for k, v in parameters.items()}
        return self.parameters
        
    def training(self, X, Y, X_test, Y_test, steps, learning_rate=0.009):
        self.initialize_parameters(X, Y)
        costs = list()
        train_accs=list()
        test_accs=list()
        try:
            for i in range(1,steps+1):
                A_L = self.propagate(X)
                cost = softmax_cost(A_L,Y)
                self.backprop(X,Y,learning_rate)
                train_acc=self.compute_accuracy(X,Y)
                test_acc=self.compute_accuracy(X_test, Y_test)
                if i%100==0 or i==1:
                    print(f"Iteration {i}: cost:{cost}; train_acc:{train_acc}; test_acc:{test_acc}") #np.round(train_acc*100)
                costs.append(cost)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        path = self.save_weights(i,train_acc,cost)
        print("Parameters saved at : ",path)
        return self.parameters, costs, train_accs, test_accs