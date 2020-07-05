# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:54:24 2020

@author: Kyle
"""

import os
from PIL import Image
import cv2
import numpy as np

from UCF_utils import sequence_generator, get_data_list

def np_to_video(array, window_name='data'):
    for frame in array:
        combined = frame.max(axis=2)
        #combined = frame[:,:,3]
        maximum = combined.max()
        normalized = abs((combined/maximum)*255)
        cv2.imshow(window_name,normalized.astype(np.ubyte))
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'data')
    dmd_dir = os.path.join(data_dir,'DMD_data')
    of_dir = os.path.join(data_dir,'OF_data')
    list_dir = os.path.join(data_dir,'ucf11TrainTestlist')
    
    '''
    num_classes= 11
    
    dmd_shape = (72,216,216,4)
    dmd_train_data, dmd_test_data, class_index = get_data_list(list_dir, dmd_dir)
    dmd_gen = sequence_generator(dmd_train_data,1,dmd_shape,num_classes)
    
    of_shape = (76,216,216,2)
    of_train_data, of_test_data, class_index = get_data_list(list_dir, of_dir)
    of_gen = sequence_generator(of_train_data,1,of_shape,num_classes)
    '''
    
    modes = np.load(os.path.join(dmd_dir, 'train/tennis_swing/v_tennis_15_08.npy'))
    np_to_video(modes,'dmd')
    