'''Execute this file in a REPL then call showImage() with an arbitrary number up to 9999 to see the corresponding mnist test image'''

import numpy as np
from mnist_loader import MnistData

from PIL import Image

def showImage(index, mnistData : MnistData):
    pixelArrays = list([pixelArray for (pixelArray, expectedResult) in mnistData.validationData])
    pixelArray = pixelArrays[index]
    scaledArray = pixelArray * 255
    scaledArray.resize((28,28))

    im = Image.fromarray(scaledArray)
    im.show()