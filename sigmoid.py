import numpy as np

def sigmoid(z :np.ndarray) -> np.ndarray:
    '''Applies the sigma function to z. (vectorized application)'''
    return 1 / (1 + np.exp(-z))


def sigmoidFirstDerivative(z :np.ndarray) -> np.ndarray:
    '''Applies the sigma first derivative function to z. (vectorized application)'''
    s = sigmoid(z)
    return s * (1 - s) 