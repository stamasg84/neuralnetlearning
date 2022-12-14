"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

class MnistData:
    def __init__(self) -> None:
        tr_d, va_d, te_d = MnistData.loadData() #Not so nice to load lots of data from file in a constructor, but that's how it will be for now

        trainingInputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        trainingResults = [MnistData.vectorizedResult(y) for y in tr_d[1]]
        self.trainingData = list(zip(trainingInputs, trainingResults)) #tuples of input vectors and expected output vectors

        validationInputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        self.validationData = list(zip(validationInputs, va_d[1])) #tuples of input vectors and expected output digits

        testInputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        self.testData = list(zip(testInputs, te_d[1])) #tuples of input vectors and expected output digits  

    @staticmethod
    def loadData():
        """Return the MNIST data as a tuple containing the training data,
        the validation data, and the test data.

        The ``training_data`` is returned as a tuple with two entries.
        The first entry contains the actual training images.  This is a
        numpy ndarray with 50,000 entries.  Each entry is, in turn, a
        numpy ndarray with 784 values, representing the 28 * 28 = 784
        pixels in a single MNIST image.

        The second entry in the ``training_data`` tuple is a numpy ndarray
        containing 50,000 entries.  Those entries are just the digit
        values (0...9) for the corresponding images contained in the first
        entry of the tuple.

        The ``validation_data`` and ``test_data`` are similar, except
        each contains only 10,000 images.

        This is a nice data format, but for use in neural networks it's
        helpful to modify the format of the ``training_data`` a little.
        That's done in the wrapper function ``load_data_wrapper()``, see
        below.
        """
        f = gzip.open('./data/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f, encoding='ISO8859-1') #without this encoding, the loading fails. the 8859-1 is just a guess....
        f.close()
        return (training_data, validation_data, test_data)    
        
    @staticmethod
    def vectorizedResult(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
