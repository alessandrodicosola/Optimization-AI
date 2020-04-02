import numpy as np
from function import Sigmoid

class NeuralNetwork:
    def __init__(self, activation_function = "sigmoid", hidden_layers = 1,nodes = 2,max_iterations=100,epsilon=1e-1,alpha=0.5,batch=10,seed = 1):
        '''
        Initialize the neural network

        Parameters
        ----------
        activation_function : Function
            Set the activation function. The default is "sigmoid".
        hidden_layers : int
            Number of hidden layers in the neural network. The default is 5.
        nodes : int
            Number of nodes for each layer. The default is 5.
        max_iterations : int
            Number of maximum iterations as stopping criteria. The default is 100.
        seed : int
            Seed for RandomState. The default is 1.

        Raises 
        ------
        RuntimeError
            Raised when activation_function is wrong.
            Values allowed = ["sigmoid"]
        '''
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self._random_state = np.random.RandomState(seed)
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.batch = batch
        self.weights = []
        self.epsilon = epsilon
        if (activation_function == "sigmoid" ):
            self.activation = Sigmoid
        else: raise RuntimeError("allowed only ['sigmoid']")

    def fit(self,X,y):
        '''
        Initialize all the parameters of the neural network
        
        N : number of observations
        D : number of attributes

        Parameters
        ----------
        X : array NxD
            Array of observations
        y : array Nx1
            Array of targets 

        Returns
        -------
        self
        '''
        
        # Add a dummy attribute for the BIAS
        _input = np.zeros((X.shape[0],X.shape[1] + 1))
        _input[:,0] = 1 
        # Set the attributes value
        _input[:,1:] = X
        
        input_layer = self._random_state.randn(_input.shape[1],self.nodes+1)
        output_layer = self._random_state.randn(self.nodes+1,len(np.unique(y)))
        hidden_layers = [self._random_state.randn(self.nodes+1,self.nodes+1) for i in range(1,self.hidden_layers)]
        
        self.weights.append(input_layer)
        for layer in hidden_layers:
            self.weights.append(layer)
        self.weights.append(output_layer)
                
        classes = np.unique(y)
        map_binary_classes = {key : [1 if class_== key else 0 for class_ in classes] for key in classes}
        
        for epoch in range(1,self.max_iterations+1):        
            
            current_X = np.copy(_input)
            current_y = np.copy(y)
            
            mse = 0
           
            # We have to empty all the input in order to go to the next epoch
            while len(current_X) > 0:
                
                current_X,current_y,selected_X,selected_y,current_batch_size = self._batch(current_X,current_y,self.batch)
               
                # Map 'layer' : [ gradient errors ]
                gradient_errors = { index : None for index in range(len(self.weights))}
                
                for observation,target in zip(selected_X,selected_y):
                    summations,activated = self._forward(observation)
                
                    binary_target = map_binary_classes[target]

                    # Backpropagate gradient error
                    for index in reversed(range(len(self.weights))):
                        if index == len(self.weights) - 1:
                            # It is the output layer
                            gradient_error = 2 * (binary_target - activated[-1]) * self.activation.derivative(summations[-1])
                        else: # It is an hidden layer
                            gradient_error = 2 * np.dot(self.weights[index + 1], gradient_error)
                            gradient_error *= self.activation.derivative(summations[index])
                            
                        # Increment the gradient for a layer
                        if gradient_errors[index] is None: gradient_errors[index]=gradient_error
                        else: gradient_errors[index] += gradient_error
                        
                # Normalize gradients
                for index in range(len(self.weights)):
                    gradient_errors[index] = gradient_errors[index] / current_batch_size
     
                # Update weights
                for index in range(len(self.weights)):
                    # if index == 0 we have to use as an input the starting input otherwise the activated summation
                    oi = np.atleast_2d(observation if index == 0 else activated[index - 1])
                    gradient = np.atleast_2d(gradient_errors[index])
                    self.weights[index] += oi.T * gradient * self.alpha
                
            # Compute the mse at the end of the epoch 
            for observation,target in zip(_input,y):
                binary_target = map_binary_classes[target]
                mse += np.mean(np.square(binary_target - self._forward(observation)[1][-1]))
            # Normalize mse of the current epoch
            mse = mse / _input.shape[0]
                
            # Print information
            if (epoch % 10 == 0): print(f"Epoch {epoch}: {mse}")
                    
            if mse < self.epsilon : return self
            
        return self
    
    
    def predict(self,X):
        '''
        Predicte the classes for each observation in X

        Parameters
        ----------
        X : array NxD
            Array of observations.

        Returns
        -------
        y : array of predictions
            Array of predictions for each observation            
        '''
        
        # Add a dummy attribute for the BIAS
        _input = np.zeros((X.shape[0],X.shape[1] + 1))
        _input[:,0] = 1 
        # Set the attributes value
        _input[:,1:] = X
        
        summations,activated = self._forward(_input)
        # Get the index where the column value is the maximum
        return (activated[-1],np.argmax(activated[-1],axis = 1))
        
    def _batch(self,X,y,size):
        '''
        Sample observations from X in batch

        size = 1 used for Pure Stochastic Gradient Descent
        size = K < N used for Batch Stochastic Gradient Descent

        Parameters
        ----------
        X : array (NxD)
            Observations.
        size : int
            Batch size. Default is 1

        Returns
        -------
        array
            Batch of observations
        '''
        batch_size = min(size,X.shape[0])
        indices = self._random_state.choice(X.shape[0],batch_size,replace=False)
        # Select X and y
        selected_X = X[indices,:]
        selected_y = y[indices]
        
        # Remove from the row the data indicated by the indices
        X = np.delete(X,indices,axis = 0)
        y = np.delete(y,indices,axis = 0)
        
        return (X,y,selected_X,selected_y,batch_size)
    
    def _forward(self,X):
        '''
        Feed forward the input
        Parameters
        ----------
        X : array (1xD)
            Single input.

        Returns
        -------
        summations : array
        activated : array
        '''
        
        
        num_layers = len(self.weights)

        summations = []
        activated = []
            
        # Summation of the input layer at the first hidden layer
        summation = np.dot(X,self.weights[0])
        summations.append(summation)
        a = self.activation.base(summation)
        a[0] = 1
        activated.append(a)
        
        # Propagation to the rest of layers
        for layer in range(1,num_layers):
            summation = np.dot(a,self.weights[layer])
            summations.append(summation)
            a = self.activation.base(summation)
            if layer != num_layers - 1: a[0]=1
            activated.append(a)
        
        
        return (summations,activated)