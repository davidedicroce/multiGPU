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
    if not os.path.exists('/dli/task/MODELS/' + expt_name):
        os.mkdir('/dli/task/MODELS/' + expt_name) 

# Path to directory containing TFRecord files
datafile = glob.glob('/dli/task/tfrecord/*')

# only set `verbose` to `1` if this is the root worker. Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0


# After N batches, will output the loss and accuracy of the last batch tested
class NBatchLogger(keras.callbacks.Callback):
    def __init__(self, display=100):
        self.seen = 0
        # Display: number of batches to wait before outputting loss
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size',0)
        if self.seen % self.display == 0:
            print('\n{}/{} - Batch Accuracy: {},  Batch Loss: {}\n'.format(self.seen, self.params['nb_sample'], self.params['metrics'][0], logs.get('loss')))

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

#train_sz = 384000*2 #qg amount in paper
#valid_sz = 12950 # qg amount in paper
#test_sz = 69653 #qg amount in paper
'''
BATCH_SZ = 32
train_sz = 32*81250
valid_sz = 32*12500
test_sz = 32*25000
'''
#'''
BATCH_SZ = 32
train_sz = 32 * 500
valid_sz = 32 * 100
test_sz = 32 * 100
#'''
valid_steps = valid_sz // BATCH_SZ
test_steps = test_sz // BATCH_SZ


channels = [0,1,2,3,4,5,6,7]
#channels = [0,1,2]
granularity=1

# Mapping functions used to convert tfrecords to tf dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
def train_dataset_generator(dataset, is_training=True, batch_sz=32, columns=[0,1,2], data_size = 32*10000):
    if is_training:
        dataset = dataset.shuffle(batch_sz * 50)

    dataset = dataset.map(map_fn).batch(batch_sz, drop_remainder=True if is_training else False)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# creates training dataset containing first data_size examples in datafile
# - datafile: name (or list of names) of TFRecord file containing training data
def get_dataset(dataset, start, end, batch_sz=32, columns=channels):
    dataset = dataset.map(map_fn).batch(batch_sz, drop_remainder=False)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def test_dataset(dataset, start, end, batch_sz=32):
    X = dataset.map(map_fn).batch(batch_sz, drop_remainder=False)
    Y = dataset.map(y_fn).batch(end-start)

    return X,Y
    
if __name__ == '__main__':
    decay = ''
    #print(">> Input file:",datafile)
    expt_name = '%s_%s'%(decay, expt_name)
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))
    
    names = datafile
    dataset = tf.data.TFRecordDataset(filenames=names, compression_type='GZIP')
    dataset = dataset.map(extract_fn)

    train_data = train_dataset_generator(dataset.take(train_sz), is_training=True, 
        batch_sz=BATCH_SZ, columns=channels, data_size=train_sz)

    val_data = get_dataset(dataset.skip(train_sz).take(valid_sz), start=train_sz, end=train_sz+valid_sz, columns=channels)
    #test_data = test_dataset(dataset.skip(train_sz+valid_sz).take(test_sz), start=train_sz+valid_sz, end=train_sz+valid_sz+test_sz)

    #from tensorflow.keras.backend import set_session
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #sess = tf.compat.v1.Session(config=config)
    #set_session(sess)
    graph = tf.compat.v1.get_default_graph()

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
    #opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-8) # changed eps to match pytorch value
    ##Scale the learning rate by the number of workers
    #opt = keras.optimizers.SGD(lr=lr_init * hvd.size(), momentum=args.momentum)
    ##use the NovoGrad optimizer instead of SGD
    opt = NovoGrad(learning_rate=lr_init * hvd.size())
    #Wrap the optimizer in a Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)
    # uses hvd.DistributedOptimizer() to compute gradients.        

    #For Horovod: We specify `experimental_run_tf_function=False` to ensure TensorFlow 
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function = False)
    #resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()
    
    #with tf.Session() as sess, graph.as_default():
    #with graph.as_default():
    #hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    #with tf.train.MonitoredTrainingSession(hooks=hook) as sess, graph.as_default():
    # Model Callbacks
    print_step = 1000
    #checkpoint = keras.callbacks.ModelCheckpoint('/uscms/home/ddicroce/work/QuarkGluon/CMSSW_9_4_17/src/QCD_Glu_Quark/MODELS/' + expt_name + '/epoch{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)#, save_weights_only=True)
    #batch_logger = NBatchLogger(display=print_step)
    #csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
    #lr_scheduler = keras.callbacks.LearningRateScheduler(LR_Decay)
    #callbacks_list=[checkpoint, csv_logger, lr_scheduler]

    callbacks_list = []
 
    #callback_list.append(NBatchLogger(display=print_step))

    #implement a LR warmup over `args.warmup_epochs`
    callbacks_list.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose))
    #replace with the Horovod learning rate scheduler, taking care not to start until after warmup is complete
    callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, multiplier=LR_Decay))
    #broadcast initial variable states from the first worker to all others by adding the broadcast global variables callback.
    callbacks_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    #average the metrics among workers at the end of every epoch by adding the metric average callback.
    callbacks_list.append(hvd.callbacks.MetricAverageCallback())  

    resume_from_epoch = 0
    #checkpointing should only be done on the root worker.
    if hvd.rank() == 0:
        callbacks_list.append(keras.callbacks.ModelCheckpoint('/dli/task/MODELS/' + expt_name + '/epoch{epoch:02d}-{val_loss:.2f}.hdf5', verbose=verbose, save_best_only=False))#, save_weights_only=True)
        callbacks_list.append(keras.callbacks.TensorBoard(args.save_dir))
    resume_from_epoch = restart_epoch(args)
    #broadcast `resume_from_epoch` from first process to all others
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0)

    history = resnet.fit(
        train_data,
        #steps_per_epoch=train_sz//BATCH_SZ,
        ##keep the total number of steps the same despite of an increased number of workers
        steps_per_epoch=train_sz // hvd.size(), 
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbose,
        workers=4,
        initial_epoch=resume_from_epoch,
        validation_data=val_data,
        #validation_steps=valid_steps,
        #set this value to be 3 * num_test_iterations / number_of_workers
        validation_steps=3 * valid_sz // hvd.size())
        #initial_epoch = args.load_epoch)
    
    #y_iter = test_data[1].make_one_shot_iterator().get_next()
    #y = sess.run(y_iter)
    #preds = resnet.predict(test_data[0], steps=test_steps, verbose=1)[:,1]
    #fpr, tpr, _ = roc_curve(y, preds)
    #roc_auc = auc(fpr, tpr)
    #print('Test AUC: ' + str(roc_auc))
    print('Network has finished training')

