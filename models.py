import numpy as np
from display import display_image
from activation_functions import *
from cost_functions import *
import json
import math

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
        dev_accs=[]
        #Iterate for every step
        for i in range(1,steps+1):
            # Define the forward propagation
            A=self.forward_propagation(X)
            cost = sigmoid_cost(A,Y)
            train_acc=self.compute_accuracy(X, Y)
            dev_acc=self.compute_accuracy(x_test,y_test)
            #Start back propagation
            self.back_propagation(A,X,Y, learning_rate)
            if i%10==0:
                #Compute the cost and the training accuracy
                print(f"Iteration {i}: cost = {cost} ; train_acc = {np.round(train_acc*100)}% ; dev_acc = {np.round(dev_acc*100)}%")
            costs.append(cost)
            train_accs.append(train_acc)
            dev_accs.append(dev_acc)
        self.save_weights(i, cost, train_acc)
        return self.parameters, costs, train_accs, dev_accs
    def save_weights(self, step, cost, train_acc):
        path = f'saved_models/binary/step_{step}_train_acc_{train_acc}_cost_{cost}.npy'
        np.save(path, self.parameters)
        return path
    def load_weights(self, path_name):
        self.parameters=np.load(f'saved_models/binary/{path_name}')
        return self.parameters
    
class multi_classification:
    def initialize_parameters(self,layers_dims):
        self.parameters=dict()
        for l in range(1, len(layers_dims)):
            self.parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
            self.parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
        return self.parameters
    def propagate(self, X):
        W1=self.parameters["W1"]
        b1=self.parameters["b1"]
        W2=self.parameters["W2"]
        b2=self.parameters["b2"]
        W3=self.parameters["W3"]
        b3=self.parameters["b3"]

        Z1=np.dot(W1,X)+b1
        A1=relu(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=relu(Z2)
        Z3=np.dot(W3,A2)+b3
        A3=softmax(Z3)

        cache={
            "Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2,
            "A3":A3
        }
        return A3, cache
    def backprop(self, X, Y, cache):
        m=X.shape[1]
        
        W2=self.parameters["W2"]
        W3=self.parameters["W3"]
        Z1=cache["Z1"]
        A1=cache["A1"]
        Z2=cache["Z2"]
        A2=cache["A2"]
        A3=cache["A3"]
        
        dZ3=A3-Y
        dW3=np.dot(dZ3,A2.T)/m
        db3=np.sum(dZ3,axis=1,keepdims=True)/m
        dZ2=np.dot(W3.T,dZ3)*(drelu(Z2))
        dW2=np.dot(dZ2,A1.T)/m
        db2=np.sum(dZ2,axis=1,keepdims=True)/m
        dZ1=np.dot(W2.T,dZ2)*(drelu(Z1))
        dW1=np.dot(dZ1,X.T)/m
        db1=np.sum(dZ1, axis=1, keepdims=True)/m
        
        grads = {
            "dW3":dW3,
            "db3":db3,
            "dW2":dW2,
            "db2":db2,
            "dW1":dW1,
            "db1":db1,
        }
        return grads
        
    def prediction(self, X):
        A_L, _ = self.propagate(X)
        return A_L
    
    def compute_accuracy(self, X, Y):
        A_L=self.prediction(X)
        num_examples = A_L.shape[1]
        max_indices = np.argmax(A_L, axis=0)
        Y_prediction = np.zeros_like(A_L)
        Y_prediction[max_indices, np.arange(num_examples)] = 1
        return np.mean(np.all(Y_prediction == Y, axis=0))
    
    def save_weights(self, epoch, train_accs, costs, dev_accs):
        path_weights=f'saved_models/multi_class/epoch_{epoch}_train_acc_{train_accs[-1]}_cost_{costs[-1]}.json'
        parameters = {k: v.tolist() for k, v in self.parameters.items()}
        with open(path_weights,'w') as json_file:
            json.dump(parameters, json_file, indent=4)
            
        path_metrics=f'saved_models/multi_class/metrics/epoch_{epoch}_train_acc_{train_accs[-1]}_cost_{costs[-1]}_metrics.json'
        metrics={
            'costs':costs,
            'train_accs':train_accs,
            'dev_accs':dev_accs
        }
        with open(path_metrics,'w') as json_file:
            json.dump(metrics, json_file, indent=4)
            
        return path_weights
    
    def load_weights(self, path):
        with open(f'saved_models/multi_class/{path}.json','r') as json_file:
            parameters=json.load(json_file)
        self.parameters={k: np.array(v) for k, v in parameters.items()}
        
        with open(f'saved_models/multi_class/metrics/{path}_metrics.json','r') as json_file:
            metrics=json.load(json_file)
        return self.parameters, metrics["costs"], metrics["train_accs"], metrics["dev_accs"]
    
    def random_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]
        mini_batches=list()
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        
        num_complete_mini_batches=math.floor(m/batch_size)
        for batch in range(0,num_complete_mini_batches):
            mini_batch_X=shuffled_X[:,batch*batch_size:(batch+1)*batch_size]
            mini_batch_Y=shuffled_Y[:,batch*batch_size:(batch+1)*batch_size]
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m%batch_size!=0:
            mini_batch_X=shuffled_X[:,(batch+1)*batch_size:]
            mini_batch_Y=shuffled_Y[:,(batch+1)*batch_size:]
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    
    def initialize_adam(self):
        L = len(self.parameters)//2
        v = dict()
        s = dict()
        for layer in range(1,L+1):
            v["dW"+str(layer)]=np.zeros((self.parameters["W" + str(layer)].shape[0],self.parameters["W" + str(layer)].shape[1]))
            v["db"+str(layer)]=np.zeros((self.parameters["b" + str(layer)].shape[0],self.parameters["b" + str(layer)].shape[1]))
            s["dW"+str(layer)]=np.zeros((self.parameters["W" + str(layer)].shape[0],self.parameters["W" + str(layer)].shape[1]))
            s["db"+str(layer)]=np.zeros((self.parameters["b" + str(layer)].shape[0],self.parameters["b" + str(layer)].shape[1]))
        return v, s            
    
    def update_parameters_adam(self,grads,v,s,t,learning_rate, beta1, beta2, epsilon):
        L = len(self.parameters)//2
        v_corrected = dict()
        s_corrected = dict()
        for layer in range(1,L+1):
            v["dW"+str(layer)]=beta1*v["dW"+str(layer)]+(1-beta1)*grads["dW" + str(layer)]
            v["db"+str(layer)]=beta1*v["db"+str(layer)]+(1-beta1)*grads["db" + str(layer)]
            
            v_corrected["dW" + str(layer)] = v["dW" + str(layer)]/(1-beta1**t)
            v_corrected["db" + str(layer)] = v["db" + str(layer)]/(1-beta1**t)
            
            s["dW"+str(layer)]=beta2*s["dW"+str(layer)]+(1-beta2)*grads["dW" + str(layer)]**2
            s["db"+str(layer)]=beta2*s["db"+str(layer)]+(1-beta2)*grads["db" + str(layer)]**2
            
            s_corrected["dW" + str(layer)] = s["dW" + str(layer)]/(1-beta2**t)
            s_corrected["db" + str(layer)] = s["db" + str(layer)]/(1-beta2**t)
            
            self.parameters["W"+str(layer)]-=learning_rate*v_corrected["dW" + str(layer)]/(np.sqrt(s_corrected["dW" + str(layer)])+epsilon)
            self.parameters["b"+str(layer)]-=learning_rate*v_corrected["db" + str(layer)]/(np.sqrt(s_corrected["db" + str(layer)])+epsilon)
        return v, s
    
    def training(self, X, Y, X_test, Y_test, epochs, learning_rate=0.009, batch_size=128, adam=True):
        np.random.seed(0)
        m=X.shape[1]
        self.initialize_parameters(layers_dims=[X.shape[0], 80, 40,Y.shape[0]])
        if adam:
            v, s = self.initialize_adam()
        costs = list()
        train_accs=list()
        dev_accs=list()
        try:
            for epoch in range(1,epochs+1):
                mini_batches = self.random_mini_batches(X, Y, batch_size)
                cost_total=0
                for t in range(1, len(mini_batches)+1):
                    batch=mini_batches[t-1]
                    Xt=batch[0]
                    Yt=batch[1]
                    A_L, cache = self.propagate(Xt)
                    cost_total += softmax_cost(A_L,Yt)
                    grads = self.backprop(Xt,Yt,cache)
                    if adam:
                        v,s=self.update_parameters_adam(grads,v,s,t,learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)
                cost_avg =cost_total/len(mini_batches)+1
                train_acc = self.compute_accuracy(X,Y)
                dev_acc=self.compute_accuracy(X_test, Y_test)
                print(f"Epoch {epoch}: cost:{cost_avg}; train_acc:{train_acc}; dev_acc:{dev_acc}") #np.round(train_acc*100)
                costs.append(cost_avg)
                train_accs.append(train_acc)
                dev_accs.append(dev_acc)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        path = self.save_weights(epoch,train_accs,costs,dev_accs)
        print("Parameters saved at : ",path)
        return self.parameters, costs, train_accs, dev_accs