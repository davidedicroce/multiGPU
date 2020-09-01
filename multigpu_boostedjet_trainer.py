import numpy as np
np.random.seed(0)
import os, glob
import time
import datetime
import tensorflow.keras as keras
import math
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

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
    args = parser.parse_args()
    
    lr_init = args.lr_init
    resblocks = args.resblocks
    epochs = args.epochs
    expt_name = 'saveBoostedJets-opendata_ResNet_blocks%d_x1_epochs%d'%(resblocks, epochs)
    expt_name = expt_name + '-' +  datetime.date.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    if len(args.name) > 0:
        expt_name = args.name
    #FIXME
    if not os.path.exists('/uscms/home/ccianfar/QCD_Glu_Quark/MODELS/' + expt_name):
        os.mkdir('/uscms/home/ccianfar/QCD_Glu_Quark/MODELS/' + expt_name) 

# Path to directory containing TFRecord files
datafile = glob.glob('/storage/local/data1/gpuscratch/ccianfar/highgrantfrecords/*')

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

#train_sz = 384000*2 #qg amount in paper
#valid_sz = 12950 # qg amount in paper
#test_sz = 69653 #qg amount in paper

BATCH_SZ = 32
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
    x = tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))[...,channels]
    y = tf.one_hot(tf.cast(data['y'], tf.uint8), classes)
    return x, y

def x_fn(data):
    return tf.reshape(data['X_jets'], (125*granularity,125*granularity,8))[...,channels]

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

    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    graph = tf.get_default_graph()

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
    opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-8) # changed eps to match pytorch value
    resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()
    
    #with tf.Session() as sess, graph.as_default():
    with graph.as_default():    
        # Model Callbacks
        print_step = 1000
        #FIXME
        checkpoint = keras.callbacks.ModelCheckpoint('/uscms/home/ccianfar/QCD_Glu_Quark/MODELS/' + expt_name + '/epoch{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)#, save_weights_only=True)
        batch_logger = NBatchLogger(display=print_step)
        csv_logger = keras.callbacks.CSVLogger('%s.log'%(expt_name), separator=',', append=False)
        lr_scheduler = keras.callbacks.LearningRateScheduler(LR_Decay)
        callbacks_list=[checkpoint, csv_logger, lr_scheduler]

        history = resnet.fit(
            train_data,
            steps_per_epoch=train_sz//BATCH_SZ,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=val_data,
            validation_steps=valid_steps,
            initial_epoch = args.load_epoch)
        
        #y_iter = test_data[1].make_one_shot_iterator().get_next()
        #y = sess.run(y_iter)
        #preds = resnet.predict(test_data[0], steps=test_steps, verbose=1)[:,1]
        #fpr, tpr, _ = roc_curve(y, preds)
        #roc_auc = auc(fpr, tpr)
        #print('Test AUC: ' + str(roc_auc))
    print('Network has finished training')

