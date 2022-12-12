import numpy as np

class Network:
    def __init__(self, sizes) -> None:
        self.sizes = sizes
        self.numLayers = len(sizes)

        rng = np.random.default_rng()

        self.biases = [rng.standard_normal(layerSize) for layerSize in sizes[1:]]
        self.weights = [rng.standard_normal((layerSize, previousLayerSize)) for previousLayerSize, layerSize in zip(sizes, sizes[1:])]

    @staticmethod
    def sigmoid(z):
        '''Applies the sigma function to z. Works if z is a numpy array too, then it returns a numpy array.'''
        return 1 / (1 + np.exp(-z))

    def feedforward(self, input):
        '''Calculates the output of the network for the specified input. Input must be a numpy array with the same length as sizes[0] was when creating the Network'''
        result = input
        for w, b in zip(self.weights, self.biases):
            result = Network.sigmoid(w@result + b)
        
        return result

sizes = [5,3,2]
a = Network(sizes)

print(a.biases)
print(a.weights)

print(a.feedforward(np.array((1, 2, 3, 4, 5))))

print('Hello GIT from private machine!')