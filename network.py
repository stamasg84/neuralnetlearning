import numpy as np
from mnist_loader import MnistData
import costfunctions
import sigmoid

class FeedforwardResult:
    def __init__(self, result : np.ndarray, activations : list[np.ndarray], zs : list[np.ndarray]) -> None:
        self.result : np.ndarray = result
        self.activations : list[np.ndarray]= activations
        self.zs : list[np.ndarray]= zs

class Network:
    def __init__(self, sizes, costFunction: costfunctions.CostFunction) -> None:
        self.sizes = sizes
        self.numLayers = len(sizes)
        self.costFunction = costFunction

        self.rng = np.random.default_rng()

        self.biases = [self.rng.standard_normal((layerSize, 1)) for layerSize in sizes[1:]]
        self.weights = [self.rng.standard_normal((layerSize, previousLayerSize)) for previousLayerSize, layerSize in zip(sizes, sizes[1:])]    

    def feedforward(self, input : np.ndarray) -> FeedforwardResult:
        '''Calculates the output of the network for the specified input and also saves the in between results
        to be able to serve as inputs to a backpropagation afterwards. Input must be a numpy array with the same length as sizes[0] was when creating the Network'''
        activations = [input]
        zs = []

        activationForLayer = input
        for w, b in zip(self.weights, self.biases):
            z = w @ activationForLayer + b
            activationForLayer = sigmoid.sigmoid(z)

            activations.append(activationForLayer)
            zs.append(z)
        
        return FeedforwardResult(activations[-1], activations, zs)

    def SGD(self, trainingInput, miniBatchSize : int, numberOfEpochs : int, eta : float, testFunction = None):
        '''Executes stochastic gradient descent on the specified training input(Tuples of training data and expected result). The sample size to use
        when calculating the cost function is miniBatchSize. An epoch consists of a weight and bias adjustment(learning) on each mini batch that comprises the trainingInput.
        In other words an epoch consumes all entries in trainingInput once, having put each of them in a miniBatch and using that miniBatch to approximate the cost function and do 
        a single weight and bias adjustment. eta is the learning rate(a greek letter) testFunction is an optional function that will be called at the end of each epoch with the network
        as a parameter. Can be used to do some evaluation of network performance'''
        for i in range(numberOfEpochs):
            shuffledTrainingInput = trainingInput.copy()
            self.rng.shuffle(shuffledTrainingInput)
            miniBatches = [shuffledTrainingInput[k:k+miniBatchSize] for k in range(0, len(trainingInput), miniBatchSize)]

            print('Learning on a total of {0} mini batches...'.format(len(miniBatches)))
            
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)            

            if testFunction:
                print('End epoch testing...')
                testFunction(self)
            
            print('Epoch {i} completed.'.format(i=i+1))
    
    def updateMiniBatch(self, miniBatch, eta):
        '''Adjusts the weights and biases of the Network by calculating the nabla weights and biases based on the specified mini batch (as opposed to calculating
        based on the complete training set)'''
        nablaWeights =[np.zeros_like(w) for w in self.weights]
        nablaBiases = [np.zeros_like(b) for b in self.biases]

        for (input, y) in miniBatch:
            deltaNablaWeights, deltaNablaBiases = self.backpropagate(input, y)
            nablaWeights = [nablaW + deltaNablaW for (nablaW, deltaNablaW) in zip(nablaWeights, deltaNablaWeights)]
            nablaBiases = [nablaB + deltaNablaB for (nablaB, deltaNablaB) in zip(nablaBiases, deltaNablaBiases)]
        
        #learning with the gradients:
        self.weights = [w - (1./len(miniBatch) * eta * nablaW) for (w, nablaW) in zip(self.weights, nablaWeights)]
        self.biases = [b - (1./len(miniBatch) * eta * nablaB) for (b, nablaB) in zip(self.biases, nablaBiases)]
       
    def backpropagate(self, input:np.ndarray, y : np.ndarray):
        '''Performs a feedforward and a backpropagation for a single input - expected output pair. Returns the derivative of the per input cost function
        for the weights and biases in a tuple'''
        feedforwardResult = self.feedforward(input)
        
        nablaWeights =[np.zeros_like(w) for w in self.weights]
        nablaBiases = [np.zeros_like(b) for b in self.biases]        

        delta = self.costFunction.Derivative(y, feedforwardResult.zs[-1]) #BP1
        nablaBiases[-1] = delta #BP3
        nablaWeights[-1] = delta @ feedforwardResult.activations[-2].transpose() #BP4

        for l in range(2, len(self.biases) + 1):
            delta = (self.weights[-l + 1].transpose() @ delta) * sigmoid.sigmoidFirstDerivative(feedforwardResult.zs[-l]) #BP2
            nablaBiases[-l] = delta #BP3
            nablaWeights[-l] = delta @ feedforwardResult.activations[-l - 1].transpose() #BP4

        return (nablaWeights, nablaBiases)

def getNetworkResultForTestData(network : Network, testData: np.ndarray):
    '''Returns the index of the neuron that yielded the highest result from the output layer of the network (for digit recognition this equals to the recognized digit)'''
    return np.argmax(network.feedforward(testData).result)

def testNetwork(network : Network, input):
    '''Assuming input is a list of tuples of inputData-expectedResult pairs, it prints a diagnostic on how many correct results the network yielded on the input'''
    resultsAndExpectations = [ (getNetworkResultForTestData(network, testData), expectedResult) for (testData, expectedResult) in input ]
    totalCorrectResults = sum(result==expectation for (result,expectation) in resultsAndExpectations)
    print('Number of correct results: {0}/{1}'.format(totalCorrectResults, len(input)))


#using the Network for digit recognition:

sizes = [784,30,10] #the input consists of images of 28*28 pixels = 784. The output layer has a neuron for each possible digit 
digitRecognizerNet = Network(sizes, costfunctions.CrossEntropyCost())

mnistData = MnistData()

digitRecognizerNet.SGD(mnistData.trainingData, 10, 30, 0.5, lambda n: testNetwork(n, mnistData.testData))