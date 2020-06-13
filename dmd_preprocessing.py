import numpy as np
from pydmd import DMD
import os
from collections import OrderedDict
import shutil
import warnings
import cv2

def dmd_prep(src_dir, dest_dir, window, svd_rank, overwrite=False):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError(dest_dir + ' already exists')
    os.mkdir(dest_dir)
    print(dest_dir, 'created')

    # create directory for training data
    dest_train_dir = os.path.join(dest_dir, 'train')
    if os.path.exists(dest_train_dir):
        print(dest_train_dir, 'already exists')
    else:
        os.mkdir(dest_train_dir)
        print(dest_train_dir, 'created')

    # create directory for testing data
    dest_test_dir = os.path.join(dest_dir, 'test')
    if os.path.exists(dest_test_dir):
        print(dest_test_dir, 'already exists')
    else:
        os.mkdir(dest_test_dir)
        print(dest_test_dir, 'created')

    dir_mapping = OrderedDict(
        [(train_dir, dest_train_dir), (test_dir, dest_test_dir)])  # the mapping between source and dest

    print('Start computing dmds ...')
    for dir, dest_dir in dir_mapping.items():
        print('Processing data in {}'.format(dir))
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class of video
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                # print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process videos one by one
                file_dir = os.path.join(class_dir, filename)
                frames = np.load(file_dir)
                # note: store the final processed data with type of float16 to save storage
                processed_data = _stack_dmd(frames, window, svd_rank).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
 

def _stack_dmd(frames, window, svd_rank, grey=True, deeper=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)i
    num_sequences = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]
    color_ch = frames.shape[3]
    
    num_modes = None
    for i in range(num_sequences - window+1):
        selection = frames[i:i+window]
        if grey:
            selection = np.array([cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in selection])
        mode = _compute_dmd(selection, svd_rank)
        if num_modes is None: 
            num_modes = mode.shape[1]
            if grey and not deeper:
                output_shape = (height,width*num_modes,num_sequences-window+1)
            elif not grey and deeper:
                output_shape = (num_sequences-window+1,height*color_ch,width,num_modes)  # dmd_modes shape is (139,968, num_modes, num_windows)
            elif grey and deeper:
                output_shape = (num_sequences-window+1,height,width,num_modes)
            else:
                output_shape = (height*color_ch,width*num_modes,num_sequences-window+1)
            modes = np.ndarray(shape=output_shape)
        if not deeper:
            mode = np.reshape(mode.T,output_shape[0:-1])
            modes[:,:,i] = mode
        else:
            mode = np.reshape(mode.T,output_shape[1:])
            modes[i] = mode
    return modes

def _compute_dmd(frames, svd_rank):
    if len(frames.shape) == 4: #if from color image
        vec_frames = np.reshape(frames, (frames.shape[0], frames.shape[1]*frames.shape[2]*frames.shape[3]))
    else: #if from greyscale image
        vec_frames = np.reshape(frames,(frames.shape[0], frames.shape[1]*frames.shape[2]))
    dmd = DMD(svd_rank=svd_rank)
    #print("input is nan: " + str(np.isnan(vec_frames).any()))
    #print("input is inf: " + str(np.isinf(vec_frames).any()))
    vec_frames /= 255.0
    dmd.fit(np.nan_to_num(vec_frames.T,posinf=255,neginf=0))
    modes = dmd.modes.real
    return modes
 
if __name__ == '__main__':
    sequence_length = 10 
    image_size = (216,216,3)
    cwd = os.getcwd()
    src_dir = os.path.join(cwd,'data/UCF-Preprocessed-DMD')
    dest_dir = os.path.join(cwd,'data/DMD_data')
    dmd_prep(src_dir, dest_dir, 5, 6, overwrite=True) 
