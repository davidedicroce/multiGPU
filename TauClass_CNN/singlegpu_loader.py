import numpy as np
np.random.seed(0)
import os, glob
import time
import datetime
import h5py
import tensorflow.keras as keras
import math
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import roc_curve, auc
from novograd import NovoGrad

from tensorflow.keras.mixed_precision import experimental as mixed_precision

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


USE_XLA = True
if USE_XLA:
    tf.config.optimizer.set_jit(USE_XLA)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

datafile = glob.glob('tfrecord_shuffle/*')

def LR_Decay(epoch):
    drop = 0.5
    epochs_drop = 10
    lr = lr_init * math.pow(drop, math.floor((epoch+1)/epochs_drop))
    return lr

BATCH_SZ    = 128
n_all       = 1000  * BATCH_SZ #SAMPLE SIZE = 3043794
train_sz    = n_all 
train_steps = 128000 // BATCH_SZ 

channels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
granularity=1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_fn(data):
    feature_description = {
        'X_jet': tf.io.FixedLenFeature([125*granularity*125*granularity*13], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    }
    sample = tf.io.parse_single_example(serialized=data, features=feature_description)
    return sample

classes = 2
def map_fn(data):
    x = tf.reshape(data['X_jet'], (125*granularity,125*granularity,13))[...,0:13]
    y = tf.zeros([125*granularity,125*granularity,13], tf.float32)
    return x, y

def x_fn(data):
    return tf.reshape(data['X_jet'], (125*granularity,125*granularity,13))[...,0:13]

def y_fn(data):
    return data['y']

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("\n Timestamp: "+str(tf.cast(tf.timestamp(),tf.float64)))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_dataset_generator(dataset, is_training=True, batch_sz=32, columns=[0,1,2], data_size = 32*10000):
    if is_training:
        print("Stage1 shuffle time: "+str(tf.cast(tf.timestamp(),tf.float64)))
        dataset = dataset.shuffle(batch_sz * 2)
        print("Stage2 shuffle time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    print("Stage1 map+shuffle+repeat+batch+prefetch time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=True if is_training else False).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    print("Stage2 map+shuffle+repeat+batch+prefetch time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    return dataset

def get_dataset(dataset, start, end, batch_sz=32, columns=channels):
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

#@nvtx_tf.ops.trace(message='test_dataset', domain_name='DataLoading', grad_domain_name='TauClass')
def test_dataset(dataset, start, end, batch_sz=32):
    X = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False)
    Y = dataset.map(y_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(end-start)
    return X,Y

def create_dataset(names):
    return tf.data.TFRecordDataset(filenames=names, compression_type='GZIP', 
            num_parallel_reads=tf.data.experimental.AUTOTUNE).map(extract_fn, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

if __name__ == '__main__':
    decay = ''

    #LOADING DATA
    dataset = create_dataset(datafile)
    train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, batch_sz=128, columns=channels, data_size=train_sz)
    
    # Build network
    inputs = tf.keras.layers.Input(shape=(125*granularity, 125*granularity, 13))
    x = tf.keras.layers.ReLU()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    opt = keras.optimizers.Adam(lr=1, epsilon=1.e-8)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    history = model.fit(
        train_data,
        steps_per_epoch=1000,  # 80000 / hvd.size()
        batch_size=128,
        epochs=1,
        workers=tf.data.experimental.AUTOTUNE,
        use_multiprocessing=True)
