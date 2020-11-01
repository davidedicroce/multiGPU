from __future__ import print_function

import os
# Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import argparse
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import fashion_mnist
from wideresnet import WideResidualNetwork
from time import time
#from novograd import NovoGrad

# Parse input arguments

parser = argparse.ArgumentParser(description='Keras Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')
parser.add_argument('--use-checkpointing', default=False, action='store_true')


args = parser.parse_args()

# Checkpoints will be written in the logs directory.
args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')

# Define a function for a simple learning rate decay over time

def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

# Define a function that tells us what epoch to restart from, if any.
# Returning 0 will mean that we want to start from scratch.

def restart_epoch(args):
    epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            epoch = try_epoch
            break

    return epoch

# Define the function that creates the model

def create_model(resume_from_epoch):
    if resume_from_epoch > 0:
        # Restore from a previous checkpoint, if initial_epoch is specified.
        model = keras.models.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
    else:
        # Set up standard WideResNet-16-10 model.
        model = WideResidualNetwork(depth=16, width=10, input_shape=input_shape,
                                    classes=num_classes, dropout_rate=0.01)

        # WideResNet model that is included with Keras is optimized for inference.
        # Add L2 weight decay & adjust BN settings.
        model_config = model.get_config()
        for layer, layer_config in zip(model.layers, model_config['layers']):
            if hasattr(layer, 'kernel_regularizer'):
                regularizer = keras.regularizers.l2(args.wd)
                layer_config['config']['kernel_regularizer'] = \
                    {'class_name': regularizer.__class__.__name__,
                     'config': regularizer.get_config()}
            if type(layer) == keras.layers.BatchNormalization:
                layer_config['config']['momentum'] = 0.9
                layer_config['config']['epsilon'] = 1e-5

        model = keras.models.Model.from_config(model_config)

        opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'],
                      experimental_run_tf_function = False) 
        
    return model

verbose = 1

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load Fashion MNIST data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Training data iterator.
train_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                     horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
train_gen.fit(x_train)
train_iter = train_gen.flow(x_train, y_train, batch_size=args.batch_size)

# Validation data iterator.
test_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.mean = train_gen.mean
test_gen.std = train_gen.std
test_iter = test_gen.flow(x_test, y_test, batch_size=args.batch_size)


callbacks = []

callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))

class PrintThroughput(keras.callbacks.Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('\nImages/sec: {}'.format(images_per_sec))

if verbose:
    callbacks.append(PrintThroughput(total_images=len(y_train)))

class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2):
        self.target = target
        self.patience = patience
        self.stopped_epoch = 0
        self.met_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.target:
            self.met_target += 1
        else:
            self.met_target = 0

        if self.met_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Early stopping after epoch {}'.format(self.stopped_epoch + 1))

callbacks.append(StopAtAccuracy(target=args.target_accuracy, patience=args.patience))

class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time after epoch {}: {}".format(epoch + 1, total_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time: {}".format(total_time))

if verbose:
    callbacks.append(PrintTotalTime())

resume_from_epoch = 0

# Create/load the model.
model = create_model(resume_from_epoch)

# Train the model.
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=resume_from_epoch,
                    validation_data=test_iter,  
                    validation_steps=len(test_iter))

# Evaluate the model on the full data set.
score = model.evaluate_generator(test_iter, len(test_iter), workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
