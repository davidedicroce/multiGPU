import numpy as np
np.random.seed(0)
import os, glob
import time
import datetime
import tensorflow.keras as keras
import math
#from tensorflow.keras.utils.io_utils import HDF5Matrix
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
    args = parser.parse_args()
    
# Path to directory containing TFRecord files
datafile = glob.glob('/home/u00u5ev76whwBTLvWe357/multiGPU/tfrecord_x1/*')

BATCH_SZ = 64
train_sz = 32*81250
valid_sz = 32*12500
test_sz = 32*25000

'''
BATCH_SZ = 32
train_sz = 32 * 500
valid_sz = 32 * 100
test_sz = 32 * 100
'''
valid_steps = valid_sz // BATCH_SZ
test_steps = test_sz // BATCH_SZ

# CHANNELS
channels = [0,1,2,3,4,5,6,7]
granularity=1

# Mapping functions used to convert tfrecords to tf dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_fn(data):
    # extracts fields from TFRecordDataset
    feature_description = {
        'X_jets': tf.FixedLenFeature([125*granularity*125*granularity*8], tf.float32),
        'm0': tf.FixedLenFeature([], tf.float32), 
        'pt': tf.FixedLenFeature([], tf.float32),
        'y': tf.FixedLenFeature([], tf.float32)
    }
    sample = tf.parse_single_example(data, feature_description)
    return sample

   
classes = 2
def map_fn(data):
    # reshapes X_jets, converts y to one-hot array for feeding into keras model
    x = tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))
    y = tf.zeros( [125*granularity, 125*granularity, 3], tf.float32)
    #y = tf.one_hot(tf.cast(data['y'], tf.uint8), classes)
    return x, y

def x_fn(data):
    return tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))

def y_fn(data):
    return data['y']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
def train_dataset_generator(dataset, is_training=True, batch_sz=32, columns=channels, data_size = 32*10000):
    if is_training:
        dataset = dataset.shuffle(batch_sz * 50)

    dataset = dataset.map(map_fn).batch(batch_sz, drop_remainder=True if is_training else False)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
def get_dataset(dataset, start, end, batch_sz=32, columns=channels):
    dataset = dataset.map(map_fn).batch(batch_sz, drop_remainder=False)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

def test_dataset(dataset, start, end, batch_sz=32):
    X = dataset.map(map_fn).batch(batch_sz, drop_remainder=False)
    Y = dataset.map(y_fn).batch(end-start)

    return X,Y

    

if __name__ == '__main__':
    
    names = datafile
    dataset = tf.data.TFRecordDataset(filenames=names, compression_type='GZIP')
    dataset = dataset.map(extract_fn)

    train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, 
        batch_sz=BATCH_SZ, columns=channels, data_size=train_sz)



    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    graph = tf.get_default_graph()


    inputs = tf.keras.layers.Input(shape=(125*granularity, 125*granularity, 8))
    x = tf.keras.layers.ReLU()(inputs)
    test = tf.keras.Model(inputs=inputs, outputs=x)

    opt = keras.optimizers.Adam(lr=1, epsilon=1.e-8)
    test.compile(loss='binary_crossentropy', optimizer=opt)


    with graph.as_default():    

        history = test.predict(
            train_data,
            steps=1600,
            verbose=1,
            max_queue_size=10
        )

