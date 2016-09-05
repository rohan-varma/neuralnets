# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

NUM_FILTERS = 20 #number of filters our convolutional network will learn.
FILTER_SIZE = 5 #the size of our filters
class LeNet:
    #build is responsible for constructing the neural net.
    #parameters: width and height of our image, number of input channels,
    #number of class labels, optional path to pre-trained model.
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()
        		# first set of CONV => RELU => POOL
		model.add(Convolution2D(NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, border_mode="same",
			input_shape=(depth, height, width)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
