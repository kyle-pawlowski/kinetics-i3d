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

#tf.compat.v1.enable_eager_execution()
num_classes = 101

def data_gen(data_folder='DMD_data',label_folder='ucfTrainTestlist'):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    list_dir = os.path.join(data_dir,label_folder)   
    video_dir = os.path.join(data_dir,data_folder)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    train_data = tf.Variable(np.array(train_data))
    test_data = tf.Variable(np.array(test_data))
    #class_index = tf.constant(class_index)
    input_shape = (216,864,6)
    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float64,tf.float64),
                                          output_shapes=((10,216,864,6),(10,101)),
                                          args=(train_data,10, input_shape, num_classes)).repeat()
    
    
#from https://www.tensorflow.org/guide/checkpoint
def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    reshaped_input = tf.reshape(example[0],(10,6,216,216,4))
    reshaped_input = tf.cast(reshaped_input,tf.float32)
    output,end_points = net(reshaped_input,is_training=True) # calls _build
    loss = tf.reduce_mean(tf.abs(output - example[1]))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss

def session_train(optimizer):
    flow_input = tf.placeholder(tf.float32,shape=(10,50,216,216,4))
    flow_answers = tf.placeholder(tf.float32,shape=(1,10,101))
    with tf.variable_scope('Flow'):
        i3d = InceptionI3d(num_classes=num_classes,spatial_squeeze=True,final_endpoint='Logits')
        logits, _ = i3d(flow_input,is_training=True)
    predictions = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.abs(predictions - flow_answers))
        
    feed_dict = {}
    data = data_gen().prefetch(1)
    iterator = data.make_initializable_iterator()
    datax,datay = iterator.get_next()
    feed_dict[flow_input] = tf.cast(tf.reshape(datax,(10,6,216,216,4)),tf.float32)
    
    global_step = tf.Variable(0,tf.int64)
    training_opt = optimizer.minimize(loss,global_step=global_step)
    is_chief = True
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
    with tf.compat.v1.train.MonitoredTrainingSession(is_chief=is_chief,hooks=[sync_replicas_hook]) as sess:
        while not sess.should_stop():
            out_logits, out_predictions = sess.run(
                [training_opt],feed_dict=feed_dict)
    
        
    
def train_network():
    #i3d = InceptionI3d(num_classes=num_classes,spatial_squeeze=False,final_endpoint='Predictions')

    # optimizers from https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SyncReplicasOptimizer
    opt = GradientDescentOptimizer(learning_rate=0.1)
    opt = SyncReplicasOptimizer(opt, replicas_to_aggregate=10,total_num_replicas=10)
    '''for example in iter(data_gen()):
        print("loop")
        train_step(i3d,example,opt)'''
    session_train(opt)
    
if __name__ is "__main__":
    train_network()