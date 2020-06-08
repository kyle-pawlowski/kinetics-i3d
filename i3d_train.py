# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:06:59 2020

@author: Pawlowski
"""
import os
import tensorflow as tf
from tensorflow.compat.v1.train import GradientDescentOptimizer, SyncReplicasOptimizer
import sonnet as snt
import numpy as np
from i3d import InceptionI3d
from UCF_utils import sequence_generator, get_data_list

tf.compat.v1.enable_eager_execution()
num_classes = 101

def data_gen(data_folder='DMD_data',label_folder='ucfTrainTestlist'):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    list_dir = os.path.join(data_dir,label_folder)   
    video_dir = os.path.join(data_dir,data_folder)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    train_data = tf.constant(np.array(train_data))
    test_data = tf.constant(np.array(test_data))
    #class_index = tf.constant(class_index)
    input_shape = (216,864,6)
    return tf.data.Dataset.from_generator(sequence_generator, (tf.float64,tf.float64),args=(train_data,76, input_shape, num_classes)).repeat()
    
    
#from https://www.tensorflow.org/guide/checkpoint
def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(tf.cast(example[0],tf.float32),True) # calls _build
    loss = tf.reduce_mean(tf.abs(output - example[1]))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss

def train_network():
    i3d = InceptionI3d(num_classes=num_classes,spatial_squeeze=False)

    # optimizers from https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SyncReplicasOptimizer
    opt = GradientDescentOptimizer(learning_rate=0.1)
    opt = SyncReplicasOptimizer(opt, replicas_to_aggregate=50,total_num_replicas=50)
    for example in iter(data_gen()):
        train_step(i3d,example,opt)
    
if __name__ is "__main__":
    train_network()