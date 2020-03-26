from abc import ABC,abstractmethod
from numpy import exp
class Function(ABC):
    def __init__(self):
        super(Function,self).__init__()
        
    @abstractmethod
    def base(self,x):
        pass
    
    @abstractmethod
    def derivative(self,x):
        pass
    
    
class Sigmoid(Function):
    @staticmethod
    def base(x):
        return 1 / (1 + exp(-x))
    @staticmethod
    def derivative(x):
        return Sigmoid.base(x)*(1 - Sigmoid.base(x))