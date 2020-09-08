import tensorflow as tf
import glob

total_images = 0
train_files = sorted(glob.glob('/home/u00u5ev76whwBTLvWe357/multiGPU/tfrecord/*'))
for f_i, file in enumerate(train_files): 
    print(f_i) 
    total_images += sum([1 for _ in tf.compat.v1.python_io.tf_record_iterator(file)])
