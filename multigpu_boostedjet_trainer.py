import numpy as np
np.random.seed(0)
import os, glob
import time
import datetime
import h5py
import tensorflow.keras as keras
import math
import tensorflow as tf
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

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Number of training epochs.')
    parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
    parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
    parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
    parser.add_argument('-a', '--load_epoch', default=0, type=int, help='Which epoch to start training from')
    parser.add_argument('-s', '--save_dir', default='MODELS', help='Directory with saved weights files')
    parser.add_argument('-n', '--name', default='', help='Name of experiment')
    parser.add_argument('--warmup-epochs', type=float, default=5, help='number of warmup epochs')
    args = parser.parse_args()
    
    lr_init = args.lr_init
    resblocks = args.resblocks
    epochs = args.epochs
    expt_name = 'BoostedJets-opendata_ResNet_blocks%d_x1_epochs%d'%(resblocks, epochs)
    expt_name = expt_name + '-' +  datetime.date.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    if len(args.name) > 0:
        expt_name = args.name
    if not os.path.exists('/home/u00u5ev76whwBTLvWe357/multiGPU/MODELS/' + expt_name):
        os.mkdir('/home/u00u5ev76whwBTLvWe357/multiGPU/MODELS/' + expt_name) 

# Path to directory containing TFRecord files
datafile = glob.glob('/home/u00u5ev76whwBTLvWe357/multiGPU/tfrecord_x1/*')

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

'''
BATCH_SZ = 32
train_sz = 32*81250
valid_sz = 32*12500
test_sz  = 32*25000
'''
#'''
BATCH_SZ = 32*50
train_sz = 32*80356
valid_sz = 32*12362
test_sz  = 32*24725
#'''
'''
BATCH_SZ = 32 * 8
train_sz = 32 * 500
valid_sz = 32 * 100
test_sz = 32 * 100
'''
valid_steps = valid_sz // (BATCH_SZ*hvd.size())
test_steps  = test_sz  // (BATCH_SZ*hvd.size())


channels = [0,1,2,3,4,5,6,7]
#channels = [0,1,2]
granularity=1

# Mapping functions used to convert tfrecords to tf dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#@nvtx_tf.ops.trace(message='ExtractFromTFRecord', domain_name='DataLoading', grad_domain_name='BoostedJets')
def extract_fn(data):
    # extracts fields from TFRecordDataset
    feature_description = {
        'X_jets': tf.io.FixedLenFeature([125*granularity*125*granularity*8], tf.float32),
        'm0': tf.io.FixedLenFeature([], tf.float32), 
        'pt': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    }
    sample = tf.io.parse_single_example(serialized=data, features=feature_description)
    return sample

classes = 2
def map_fn(data):
    # reshapes X_jets, converts y to one-hot array for feeding into keras model
    x = tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))[...,0:8]
    y = tf.one_hot(tf.cast(data['y'], tf.uint8), classes)
    return x, y

def x_fn(data):
    return tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))[...,0:8]

def y_fn(data):
    return data['y']

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("\n Timestamp: "+str(tf.cast(tf.timestamp(),tf.float64)))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
#@nvtx_tf.ops.trace(message='train_dataset', domain_name='DataLoading', grad_domain_name='BoostedJets')
def train_dataset_generator(dataset, is_training=True, batch_sz=32, columns=[0,1,2], data_size = 32*10000):
    if is_training:
        dataset = dataset.shuffle(batch_sz * 50)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=True if is_training else False).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(map_fn,num_parallel_calls=NUM_WORKERS).batch(batch_sz, drop_remainder=True if is_training else False)
    # dataset = dataset.repeat()
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
#@nvtx_tf.ops.trace(message='get_dataset', domain_name='DataLoading', grad_domain_name='BoostedJets')
def get_dataset(dataset, start, end, batch_sz=32, columns=channels):
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(map_fn, num_parallel_calls=NUM_WORKERS).batch(batch_sz, drop_remainder=False)
    #dataset = dataset.repeat()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset

#@nvtx_tf.ops.trace(message='test_dataset', domain_name='DataLoading', grad_domain_name='BoostedJets')
def test_dataset(dataset, start, end, batch_sz=32):
    X = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_sz, drop_remainder=False)
    #X = dataset.map(map_fn).batch(batch_sz, drop_remainder=False)
    Y = dataset.map(y_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(end-start)
    #Y = dataset.map(y_fn).batch(end-start)

    return X,Y

#@nvtx_tf.ops.trace(message='Creat Resnet', domain_name='Resnet', grad_domain_name='BoostedJets')
def create_resnet():
    # Build network
    import keras_resnet_single as networks
    resnet = networks.ResNet.build(len(channels), resblocks, [16,32], (125*granularity,125*granularity,len(channels)), granularity)
    # Load saved weights, if indicated
    if args.load_epoch != 0:
        directory = args.save_dir
        if args.save_dir == '':
            directory = expt_name
        model_name = glob.glob('MODELS/%s/epoch%02d-*.hdf5'%(directory, args.load_epoch))[0]
        #assert len(model_name) == 2
        #model_name = model_name[0].split('.hdf5')[0]+'.hdf5'
        print('Loading weights from file:', model_name)
        resnet.load_weights(model_name)
    #opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-5) # changed eps to match pytorch value
    #opt = keras.optimizers.SGD(lr=lr_init * hvd.size())
    opt = NovoGrad(learning_rate=lr_init * hvd.size())
    #Wrap the optimizer in a Horovod distributed optimizer -> uses hvd.DistributedOptimizer() to compute gradients.
    opt = hvd.DistributedOptimizer(opt)

    #For Horovod: We specify `experimental_run_tf_function=False` to ensure TensorFlow
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function = False)
    #resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()
    return resnet

#def pin_memory(self):
#    self.inp = self.inp.pin_memory()
#    self.tgt = self.tgt.pin_memory()
#    return self
    
if __name__ == '__main__':
    decay = ''
    #print(">> Input file:",datafile)
    expt_name = '%s_%s'%(decay, expt_name)
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    
    # Build network
    resnet = create_resnet()

    # Model Callbacks
    callbacks_list = []
    callbacks_list.append(myCallback())
    callbacks_list.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose))
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, multiplier=LR_Decay))
    callbacks_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks_list.append(hvd.callbacks.MetricAverageCallback())  
    resume_from_epoch = 0
    #checkpointing should only be done on the root worker.
    if hvd.rank() == 0:
        callbacks_list.append(keras.callbacks.ModelCheckpoint('/home/u00u5ev76whwBTLvWe357/multiGPU/MODELS/' + expt_name + '/epoch{epoch:02d}-{val_loss:.2f}.hdf5', verbose=verbose, save_best_only=False))#, save_weights_only=True)
        callbacks_list.append(keras.callbacks.TensorBoard(args.save_dir))
    resume_from_epoch = restart_epoch(args)
    #broadcast `resume_from_epoch` from first process to all others
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0)

    #LOADING DATA
    names = datafile
    dataset = tf.data.TFRecordDataset(filenames=names, compression_type='GZIP', num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(extract_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = tf.data.TFRecordDataset(filenames=names, compression_type='GZIP', buffer_size=100, num_parallel_reads=4)
    #dataset = tf.data.TFRecordDataset(filenames=names, compression_type='GZIP')
    #dataset = dataset.map(extract_fn)

    #train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, batch_sz=BATCH_SZ, columns=channels, data_size=train_sz, pin_memory = True)
    train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, batch_sz=BATCH_SZ, columns=channels, data_size=train_sz)

    val_data = get_dataset(dataset.skip(train_sz).take(valid_sz), start=train_sz, end=train_sz+valid_sz, columns=channels)
    #test_data = test_dataset(dataset.skip(train_sz+valid_sz).take(test_sz), start=train_sz+valid_sz, end=train_sz+valid_sz+test_sz)

    history = resnet.fit(
        train_data,
        steps_per_epoch= train_sz // (BATCH_SZ*hvd.size()), 
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbose,
        workers=hvd.size(),
        initial_epoch=resume_from_epoch,
        validation_data=val_data,
        validation_steps = 3 * valid_steps)
        #initial_epoch = args.load_epoch)
    
    #y_iter = test_data[1].make_one_shot_iterator().get_next()
    #y = sess.run(y_iter)
    #preds = resnet.predict(test_data[0], steps=test_steps, verbose=1)[:,1]
    #fpr, tpr, _ = roc_curve(y, preds)
    #roc_auc = auc(fpr, tpr)
    #print('Test AUC: ' + str(roc_auc))
    print('Network has finished training')

