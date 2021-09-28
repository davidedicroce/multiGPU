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

#import nvtx.plugins.tf as nvtx_tf

#import Horovod
import horovod.tensorflow.keras as hvd
#initialize Horovod
hvd.init()
#pin to a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

USE_XLA = True
if USE_XLA:
    tf.config.optimizer.set_jit(USE_XLA)
    #reference url : https://www.tensorflow.org/xla

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#class TimingCallback(tf.keras.callbacks.Callback):
#    def __init__(self):
#        self.logs=[]
#
#    def on_epoch_begin(self, epoch, logs={}):
#        self.starttime=time()
#
#    def on_epoch_end(self, epoch, logs={}):
#        time_interval = time()-self.starttime
#        print("Time taken for epoch {} : {}".format(epoch, time_interval))
#        self.logs.append(time_interval)    
#
## Profiling
#print("TensorFlow version: ", tf.__version__)
#
#if hvd.local_rank() == 0:
#   device_name = tf.test.gpu_device_name()
#   if not device_name:
#       raise SystemError('GPU device not found')
#       print('Found GPU at: {}'.format(device_name))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
    parser.add_argument('-b', '--resblocks', default=5, type=int, help='Number of residual blocks.')
    parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
    parser.add_argument('-a', '--load_epoch', default=0, type=int, help='Which epoch to start training from')
    parser.add_argument('-s', '--save_dir', default='MODELS', help='Directory with saved weights files')
    parser.add_argument('-n', '--name', default='', help='Name of experiment')
    parser.add_argument('--warmup-epochs', type=float, default=2, help='number of warmup epochs')
    args = parser.parse_args()
    
    lr_init = args.lr_init
    resblocks = args.resblocks
    epochs = args.epochs
    expt_name = 'TauClass-opendata_ResNet_blocks%d_x1_epochs%d_Adam_v3'%(resblocks, epochs)
    expt_name = expt_name + '-' +  datetime.date.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    if len(args.name) > 0:
        expt_name = args.name
    if not os.path.exists('MODELS/' + expt_name):
        os.mkdir('MODELS/' + expt_name) 

# Path to directory containing TFRecord files
datafile = glob.glob('tfrecord_shuffle/*')

# only set `verbose` to `1` if this is the root worker. Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0


def LR_Decay(epoch):
    drop = 0.5
    epochs_drop = 10
    lr = lr_init * math.pow(drop, math.floor((epoch+1)/epochs_drop))
    return lr

def restart_epoch(args):
    epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(expt_name.format(epoch=try_epoch)):
            epoch = try_epoch
            break

    return epoch

BATCH_SZ    = 128
n_steps     = 3751867 // BATCH_SZ
#n_steps     = 3721855 // BATCH_SZ
n_all       = n_steps * BATCH_SZ #SAMPLE SIZE = 3043794
valid_sz    = n_all // 10         #(20% for validation)
test_sz     = n_all // 10
valid_steps = valid_sz // BATCH_SZ
test_steps  = test_sz // BATCH_SZ
train_sz    = BATCH_SZ * (n_steps - valid_steps - test_steps)
train_steps = train_sz // BATCH_SZ 

print("Sample size: ", n_all, " (",n_steps, " steps), Train: ", train_sz, ", Validation: ", valid_sz)

channels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
granularity=1

# Mapping functions used to convert tfrecords to tf dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#@nvtx_tf.ops.trace(message='ExtractFromTFRecord', domain_name='DataLoading', grad_domain_name='TauClass')
def extract_fn(data):
    # extracts fields from TFRecordDataset
    feature_description = {
        'X_jet': tf.io.FixedLenFeature([125*granularity*125*granularity*13], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    }
    sample = tf.io.parse_single_example(serialized=data, features=feature_description)
    return sample

classes = 1
def map_fn(data):
    # reshapes X_jet, converts y to one-hot array for feeding into keras model
    x = tf.reshape(data['X_jet'], (125*granularity,125*granularity,13))[...,0:13]
    y = tf.cast(data['y'], tf.uint8)
    return x, y

def x_fn(data):
    return tf.reshape(data['X_jet'], (125*granularity,125*granularity,13))[...,0:13]

def y_fn(data):
    return data['y']

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("\n Timestamp: "+str(tf.cast(tf.timestamp(),tf.float64)))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
#@nvtx_tf.ops.trace(message='train_dataset', domain_name='DataLoading', grad_domain_name='TauClass')
def train_dataset_generator(dataset, is_training=True, batch_sz=32, columns=[0,1,2], data_size = 32*10000):
    if is_training:
        print("Stage1 shuffle time: "+str(tf.cast(tf.timestamp(),tf.float64)))
        dataset = dataset.shuffle(batch_sz * 2)
        print("Stage2 shuffle time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    print("Stage1 map+shuffle+repeat+batch+prefetch time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=True if is_training else False).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    print("Stage2 map+shuffle+repeat+batch+prefetch time: "+str(tf.cast(tf.timestamp(),tf.float64)))
    return dataset

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
#@nvtx_tf.ops.trace(message='get_dataset', domain_name='DataLoading', grad_domain_name='TauClass')
def get_dataset(dataset, start, end, batch_sz=32, columns=channels):
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False).repeat().prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

#@nvtx_tf.ops.trace(message='test_dataset', domain_name='DataLoading', grad_domain_name='TauClass')
def test_dataset(dataset, start, end, batch_sz=32):
    X = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False)
    Y = dataset.map(y_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(end-start)

    return X,Y

#@nvtx_tf.ops.trace(message='Creat Resnet', domain_name='Resnet', grad_domain_name='TauClass')
def create_resnet():
    # Build network
    import keras_resnet_v3 as networks
    resnet = networks.ResNet.build(len(channels), resblocks, [32,64], (125*granularity,125*granularity,len(channels)), granularity)
    # Load saved weights, if indicated
    if args.load_epoch != 0:
        directory = args.save_dir
        if args.save_dir == '':
            directory = expt_name
        print('MODELS/%s/epoch%02d'%(directory, args.load_epoch))
        model_name = glob.glob('MODELS/%s/epoch%02d*.hdf5'%(directory, args.load_epoch))[0]
        print('Loading weights from file:', model_name)
        resnet.load_weights(model_name)
    opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-8) # changed eps to match pytorch value
    #opt = NovoGrad(learning_rate=lr_init * hvd.size())
    #radam = tfa.optimizers.RectifiedAdam()
    #ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    #Wrap the optimizer in a Horovod distributed optimizer -> uses hvd.DistributedOptimizer() to compute gradients.
    opt = hvd.DistributedOptimizer(opt)

    #For Horovod: We specify `experimental_run_tf_function=False` to ensure TensorFlow
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function = False)
    if hvd.rank() == 0:
        resnet.summary()
    #resnet.summary()
    return resnet

def create_dataset(names):
    return tf.data.TFRecordDataset(filenames=names, compression_type='GZIP', 
            num_parallel_reads=tf.data.experimental.AUTOTUNE).map(extract_fn, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

if __name__ == '__main__':
    decay = ''
    for d in ['MODELS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))

    #LOADING DATA
    names = datafile
    dataset = create_dataset(names)
    train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, batch_sz=BATCH_SZ, columns=channels, data_size=train_sz)
    val_data = get_dataset(dataset.skip(train_sz).take(valid_sz), start=train_sz, end=train_sz+valid_sz, columns=channels)
    test_data = test_dataset(dataset.skip(train_sz+valid_sz).take(test_sz), start=train_sz+valid_sz, end=train_sz+valid_sz+test_sz)
    
    # Build network
    resnet = create_resnet()

    # Model Callbacks
    initial_lr=lr_init
    callbacks_list = []
    callbacks_list.append(myCallback())
    allbacks_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks_list.append(hvd.callbacks.MetricAverageCallback())
    callbacks_list.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs,verbose=verbose if hvd.rank()==0 else 0,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=10, multiplier=1.,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=10, end_epoch=11, multiplier=2e-1,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=11, end_epoch=12, multiplier=1e-1,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=12, end_epoch=13, multiplier=2e-2,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=13, end_epoch=14, multiplier=1e-2,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=14, end_epoch=15, multiplier=2e-3,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=16, multiplier=1e-3,initial_lr=lr_init))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=16, multiplier=1e-4,initial_lr=lr_init))

    # Horovod: write logs on worker 0.
    #if hvd.rank() == 0:
    #    logs = "METRICS/%s"%(expt_name)

    #checkpointing should only be done on the root worker.
    if hvd.rank() == 0:
        callbacks_list.append(keras.callbacks.ModelCheckpoint('./MODELS/' + expt_name + '/epoch{epoch:02d}-{val_accuracy:.3f}-{val_loss:.3f}.hdf5', verbose=verbose, save_best_only=False))#, save_weights_only=True)
        callbacks_list.append(keras.callbacks.TensorBoard(args.save_dir))
        callbacks_list.append(keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=True))
    resume_from_epoch = args.load_epoch
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0)

    history = resnet.fit(
        train_data,
        steps_per_epoch=train_steps,  # 80000 / hvd.size()
        batch_size=BATCH_SZ,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbose if hvd.rank()==0 else 0,
        workers=tf.data.experimental.AUTOTUNE,
        use_multiprocessing=True,
        initial_epoch=resume_from_epoch,
        validation_data=val_data,
        validation_steps = valid_steps)
    
    print('Network has finished training')
    resnet.save('resnet_v2.hdf5')

    print("Running Inference")
    pred = resnet.predict(test_data[0],batch_size=BATCH_SZ,verbose=verbose if hvd.rank()==0 else 0, workers=tf.data.experimental.AUTOTUNE)
    fpr, tpr, _ = roc_curve(np.squeeze(np.array(list(test_data[1].as_numpy_iterator()))),np.array(pred))
    roc_auc = auc(fpr, tpr)
    print('Test AUC: ' + str(roc_auc))

