#import keras
#from keras.models import Sequential
#from keras import layers
#from keras.layers.merge import add

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



def ResBlock(in_channels, out_channels):

    def f(x):
        residual = x

        downsample = out_channels//in_channels

        conv = layers.Conv2D(out_channels, activation='relu', kernel_size=(3,3), strides=(downsample,downsample), padding='SAME')(x)
        #conv = layers.Conv2D(out_channels, input_shape=keras.backend.shape(x)[1:], activation='relu', kernel_size=(3,3), strides=(downsample,downsample), padding='SAME')(x)
        #conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2D(out_channels, kernel_size=(3,3), padding='SAME')(conv)
        #conv = layers.BatchNormalization()(conv)

        if downsample > 1:
            residual = layers.Conv2D(out_channels, kernel_size=1, strides=downsample)(x)

        block = layers.Add()([conv, residual])
        block = layers.Activation('relu')(block)

        return block
    return f

#ResBlocks
def block_layers(nblocks, fmaps):
    def f(x):
        for _ in range(nblocks):
            x = ResBlock(fmaps[0], fmaps[1])(x)
        return x
    return f


class ResNet(object):

    @staticmethod
    def build(in_channels, nblocks, fmaps, input_shape=(125,125,3), gran=1):
        input = layers.Input(shape=input_shape)

        #conv0 - changed padding from 1 to 'SAME'
        #x = layers.Conv2D(fmaps[0], input_shape=input_shape, kernel_size=(7,7), strides=(2,2), padding='SAME')(input)
        x = layers.Conv2D(fmaps[0], input_shape=input_shape, kernel_size=(gran*7,gran*7), strides=(gran*2,gran*2), padding='SAME')(input)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

        x = block_layers(nblocks, [fmaps[0],fmaps[0]])(x)
        x = block_layers(1, [fmaps[0],fmaps[1]])(x)
        x = block_layers(nblocks, [fmaps[1],fmaps[1]])(x)

        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        predictions = layers.Dense(2)(x)
        predictions = layers.Activation('sigmoid')(predictions)

        model = keras.Model(inputs=input, outputs=predictions)
        return model

