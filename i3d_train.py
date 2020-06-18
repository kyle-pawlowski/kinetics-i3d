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
import datetime

#tf.compat.v1.enable_eager_execution()
num_classes = 101
batch_size = 10

def data_gen(data_folder='DMD_data',label_folder='ucfTrainTestlist'):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    list_dir = os.path.join(data_dir,label_folder)   
    video_dir = os.path.join(data_dir,data_folder)
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    train_data = tf.constant(np.array(train_data))
    test_data = tf.constant(np.array(test_data))
    n = tf.constant(num_classes)
    batch_size = tf.constant(10)
    #class_index = tf.constant(class_index)
    input_shape = (12,216,216,4)
    return tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.float32,tf.float32),
                                          output_shapes=(tf.TensorShape([10,12,216,216,4]),tf.TensorShape([10,101])),
                                          args=(train_data,batch_size, input_shape, n))
    
    
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

def session_train(optimizer,epochs):
    #build network with fake inputs
    flow_input = tf.placeholder(tf.float32,shape=tf.TensorShape([10,12,216,216,4]))
    flow_answers = tf.placeholder(tf.float32,shape=(10,101))
    with tf.variable_scope('Flow'):
        i3d = InceptionI3d(num_classes=num_classes,spatial_squeeze=True,final_endpoint='Logits')
        logits, _ = i3d(flow_input,is_training=True)
    predictions = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.abs(predictions - flow_answers))
    maxes = tf.math.reduce_max(predictions,axis=[1],keepdims=True)
    guesses = tf.math.equal(predictions,maxes)
    correct = tf.math.logical_and(guesses,tf.math.equal(flow_answers,tf.constant(1,dtype=tf.float32)))
    correct = tf.cast(correct,tf.uint8)
    accuracy = (tf.math.reduce_sum(correct)/batch_size)*100
    
    
    #create data iterator
    data = data_gen()
    iterator = tf.compat.v1.data.make_one_shot_iterator(data)
    datax,datay = iterator.get_next()
    datax=tf.cast(datax,tf.float32)
    datay=tf.cast(datay,tf.float32)
    
    #create folders for summary logs
    # https://www.tensorflow.org/tensorboard/get_started
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/training' + time
    train_summary_writer = tf.compat.v1.summary.FileWriter(train_log_dir)
    
    #create variable map to save like in evaluate sample
    variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        variable_map[variable.name.replace(':0', '')] = variable
    
    #create checkpoint 
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/Checkpoint
    #savior = tf.train.Saver(var_list=variable_map, reshape=True)
    #ckpt = tf.train.Checkpoint(optimizer=optimizer, model=i3d)
    #manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    
    #evaluation
    global_step = tf.compat.v1.train.get_or_create_global_step()
    step_counter_hook = tf.estimator.StepCounterHook(summary_writer=train_summary_writer)
    training_opt = optimizer.minimize(loss,global_step=tf.compat.v1.train.get_global_step())
    is_chief = True
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
    with tf.compat.v1.train.MonitoredTrainingSession(is_chief=is_chief,
                                                     hooks=[sync_replicas_hook,step_counter_hook],
                                                     checkpoint_dir='./tf_ckpts',
                                                     summary_dir='./logs',
                                                     save_summaries_steps=1,
                                                    save_checkpoint_steps=2) as sess:
        for epoch in range(epochs):
            feed_dict = {}
            feed_dict[flow_input] = sess.run(datax)
            feed_dict[flow_answers] = sess.run(datay)
            epoch_loss, epoch_accuracy, result = sess.run([loss,accuracy,training_opt],feed_dict=feed_dict)
            print('loss: '+str(epoch_loss))
            print('accuracy: ' + str(accuracy))
            #ckpt.step.assign_add(1)
            #if epoch % 2 == 0:
             # save_path = savior.save(sess, 'my_model', global_step=global_step)
    
def train_network():
    #i3d = InceptionI3d(num_classes=num_classes,spatial_squeeze=False,final_endpoint='Predictions')

    # optimizers from https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SyncReplicasOptimizer
    opt = GradientDescentOptimizer(learning_rate=0.1)
    opt = SyncReplicasOptimizer(opt, replicas_to_aggregate=1,total_num_replicas=1)
    '''for example in iter(data_gen()):
        print("loop")
        train_step(i3d,example,opt)'''
    session_train(opt,2)
    
if __name__ is "__main__":
    train_network()