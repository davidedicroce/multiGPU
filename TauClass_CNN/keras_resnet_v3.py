import torch
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

class Temperature(keras.layers.Layer):
  def __init__(self):
    super(Temperature, self).__init__()
    self.temperature = torch.nn.Parameter(torch.ones(1))

  def call(self, final_output):
    return final_output/ self.temperature


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

        layers.Dropout(0.01),

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

        x = layers.Dense(128, activation='relu')(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=input, outputs=predictions)
        return model


