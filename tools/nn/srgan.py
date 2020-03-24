from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape

# SRGAN
class SRGAN:
    @staticmethod
    def generator(input_shape):
        # Input shape like: (64, 64, 3)
        momentum = 0.8
        input_layer = Input(shape=input_shape)

		# First piece of the net
        conv1 = Conv2D(filters=64, kernel_size=9, strides=1, padding="same",
        activation="prelu")

        # create residual blocks


        # Elementwise sum of conv1 and post resblock conv2


		# return the generator model
		return **
    
    def res_blocks(x):

        # first res block
        res = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(x)
        res = Activation('prelu')(res)
        res = BatchNormalization(momentum=0.8)(res)
        res = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(res)
        res = BatchNormalization(momentum=0.8)(res)

        # elementwise sum
        res = Add()(res, x)

        return res
                     