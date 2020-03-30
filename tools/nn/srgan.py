from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras.layers import UpSampling2D
from keras.layers import Add
from keras import Input, Model
from keras.applications import VGG19

# SRGAN
class SRGAN:

    @staticmethod
    def generator(input_shape):
        # Input shape like: (64, 64, 3)
        input_layer = Input(shape=input_shape)

        # define function to create residual blocks
        def res_block(x):

            # first res block
            res = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(x)
            res = BatchNormalization(momentum=0.8)(res)
            res = PReLU()(res)
            res = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(res)
            res = BatchNormalization(momentum=0.8)(res)

            # elementwise sum
            res = Add()([res, x])

            return res

		# First piece of the net
        conv1 = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(input_layer)
        conv1 = PReLU()(conv1)

        # create residual blocks
        blocks = res_block(conv1)
        for i in range(0, 15):
            blocks = res_block(blocks)


        # Elementwise sum of conv1 and post resblock conv
        conv2 = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(blocks)
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = Add()([conv1, conv2])

        # third conv block
        conv3 = Conv2D(kernel_size=3, filters=256, strides=1, padding="same")(conv2)
        conv3 = UpSampling2D(size = 2)(conv3)
        conv3 = PReLU()(conv3)

        # fourth block - same as the third one
        conv4 = Conv2D(kernel_size=3, filters=256, strides=1, padding="same")(conv3)
        conv4 = UpSampling2D(size = 2)(conv4)
        conv4 = PReLU()(conv4)

        # output conv
        out = Conv2D(kernel_size=9, filters=3, strides=1, padding="same")(conv4)
        out = Activation('tanh')(out)

        # create model
        model = Model(inputs=[input_layer], output=[out], name="generator")

        # return model
        return model

    @staticmethod
    def discriminator(input_shape):

        # Input shape like: (64, 64, 3)
        input_layer = Input(shape=input_shape)

        # Initial block
        conv = Conv2D(kernel_size=3, filters=64, strides=1, padding="same")(input_layer)
        conv = LeakyReLU(alpha=0.2)(conv)

        # configuration for the next conv blocks
        config = [
            {'filters':64, 'kernel':3, 'strides':2},
            {'filters':128, 'kernel':3, 'strides':1},
            {'filters':128, 'kernel':3, 'strides':2},
            {'filters':256, 'kernel':3, 'strides':1},
            {'filters':256, 'kernel':3, 'strides':2},
            {'filters':512, 'kernel':3, 'strides':1},
            {'filters':512, 'kernel':3, 'strides':2}
        ]

        # create next few blocks
        for i in range (7):

            conv = Conv2D(kernel_size=config[i]['kernel'],
                filters=config[i]['filters'], strides=config[i]['strides'],
                padding="same")(conv)
            conv = BatchNormalization(momentum=0.8)(conv)
            conv = LeakyReLU(alpha=0.2)(conv)

        #  Dense layer 1
        dense1 = Dense(units=1024)(conv)
        dense1 = LeakyReLU(alpha=0.2)(dense1)

        # last dense and output
        output = Dense(units=1, activation='sigmoid')(dense1)

        model = Model(inputs=[input_layer], outputs=[output], name="discriminator")

        return model
    
    @staticmethod
    def build(input_shape, generator, discriminator, vgg):
        # Input shape like: (64, 64, 3)

        input_layer = Input(shape=input_shape)
        output_shape = Input(shape=(128, 128, 3))

        # generator
        generated_img = generator(input_layer)

        # extract generator features using VGG
        features = vgg(generated_img)

        # discriminator
        output = discriminator(generated_img)

        # create adversial model
        model = Model([input_layer, output_shape], [output, features])

        # print summary of the model
        for layer in model.layers:
            print(layer.name, layer.trainable)
        
        print(model.summary())

        # return model
        return model

    @staticmethod
    def vgg(input_shape):

        input_layer = Input(shape=input_shape)

        vgg = VGG19(include_top=False, weights='imagenet')

        features = vgg(input_layer)

        model = Model(inputs=[input_layer], outputs=[features])
        model.trainable = False

        return model

