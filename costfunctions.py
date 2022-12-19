import sigmoid
import numpy as np

class CostFunction:
    def Derivative(self, y : np.array , z: np.array):
        pass

class CrossEntropyCost(CostFunction):    
    def Derivative(self, y : np.array , z: np.array):
        '''Returns the delta vector for a single x from the input with y expected result and z weighted cost'''
        return sigmoid.sigmoid(z) - y

class QuadraticCost(CostFunction):
    def Derivative(self,y : np.array , z: np.array):
        '''Returns the delta vector for a single x from the input with y expected result and z weighted cost'''
        return (sigmoid.sigmoid(z) - y) * sigmoid.sigmoidFirstDerivative(z)