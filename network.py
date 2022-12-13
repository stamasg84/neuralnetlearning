import numpy as np
from mnist_loader import MnistData

class Network:
    def __init__(self, sizes) -> None:
        self.sizes = sizes
        self.numLayers = len(sizes)

        self.rng = np.random.default_rng()

        self.biases = [self.rng.standard_normal(layerSize) for layerSize in sizes[1:]]
        self.weights = [self.rng.standard_normal((layerSize, previousLayerSize)) for previousLayerSize, layerSize in zip(sizes, sizes[1:])]

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

    def SGD(self, trainingInput, miniBatchSize, numberOfEpochs, eta, testFunction = None):
        '''Executes stochastic gradient descent on the specified training input(Tuples of training data and expected result). The sample size to use
        when calculating the cost function is miniBatchSize. An epoch consists of a weight and bias adjustment(learning) on each mini batch that comprises the trainingInput.
        In other words an epoch consumes all entries in trainingInput once, having put each of them in a miniBatch and using that miniBatch to approximate the cost function and do 
        a single weight and bias adjustment. eta is the learning rate(a greek letter) testFunction is an optional function that will be called at the end of each epoch with the network
        as a parameter. Can be used to do some evaluation of network performance'''
        for i in range(numberOfEpochs):
            shuffledTrainingInput = trainingInput.copy()
            self.rng.shuffle(shuffledTrainingInput)
            miniBatches = [shuffledTrainingInput[k:k+miniBatchSize] for k in range(0, len(trainingInput), miniBatchSize)]

            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            
            if testFunction:
                print('Test result for epoch:')
                testFunction(self)
            
            print('Epoch {i} completed.'.format(i=i+1))
    
    def updateMiniBatch(self, miniBatch, eta):
        print(miniBatch)
        

def getNetworkResultForTestData(network, testData):
    '''Returns the index of the neuron that yielded the highest result from the output layer of the network (for digit recognition this equals to the recognized digit)'''
    return np.argmax(network.feedforward(testData))

def testNetwork(network, input):
    '''Assuming input is a list of tuples of inputData-expectedResult pairs, it prints a diagnostic on how many correct results the network yielded on the input'''
    resultsAndExpectations = [ (getNetworkResultForTestData(network, testData), expectedResult) for (testData, expectedResult) in input ]
    totalCorrectResults = sum(result==expectation for (result,expectation) in resultsAndExpectations)
    print('Number of correct results:', totalCorrectResults)


sizes = [3,3,2]
digitRecognizerNet = Network(sizes)

mnistData = MnistData()

trainingInput = [(np.array((1, 2, 3)), 1), 
                 (np.array((3, 4, 5)), 9), 
                 (np.array((2, 2, 2)), 7), 
                 (np.array((6, 1, 1)), 2), 
                 (np.array((1, 2, 5)), 2), 
                 (np.array((3, 4, 5)), 9), 
                 (np.array((1, 2, 3)), 1), 
                 (np.array((3, 4, 5)), 9)]

testData = [ (np.array((1, 2, 3)), 1), (np.array((3, 4, 5)), 9) ]

digitRecognizerNet.SGD(trainingInput, 2, 3, 0.1, lambda n: testNetwork(n, testData))