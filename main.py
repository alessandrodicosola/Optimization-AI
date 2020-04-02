# Import sklearn in order to be able to use utility for testing the **NeuralNetwork**
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Defining my neural network
from neural_network import NeuralNetwork

def accuracy_score(y_true,y_predict):
    return (list(y_true==y_predict)).count(True)/len(y_predict)


X,y = make_classification(n_samples=200, n_features=4, n_redundant=0,
                        n_informative=1, random_state=1,
                        n_clusters_per_class=1,n_classes = 2)

X_train,X_test,y_train,y_test = train_test_split(X,y)


  
# Fit and predict the data
hidden_layers = 5
nodes = 5
network = NeuralNetwork("sigmoid",hidden_layers = hidden_layers, nodes = nodes)
output_layer, y_predict = network.fit(X_train,y_train).predict(X_test)


print("Accuracy test: ", accuracy_score(y_test, y_predict))
    