# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:17:01 2020

@author: Pawlowski
"""

import os
import shutil
cwd = os.getcwd()
data_dir = os.path.join(cwd,'data')
list_dir = os.path.join(data_dir,'ucf11TrainTestlist')
video_dir = os.path.join(data_dir,'UCF11')
folders = os.listdir(video_dir)
i=0
classnum=0
with open(os.path.join(list_dir,'testlist.txt'),'w') as test_splits:
    with open(os.path.join(list_dir,'trainlist.txt'),'w') as train_splits:
        with open(os.path.join(list_dir,'classInd.txt'),'w') as classInd:
            for folder in folders:
                shutil.rmtree(os.path.join(video_dir,folder,'Annotation'))
                files = os.listdir(os.path.join(video_dir,folder))
                for file in files:
                    subfiles = os.listdir(os.path.join(video_dir,folder,file))
                    for subfile in subfiles:
                        if i%4 == 0:
                            test_splits.write(folder+'/'+subfile+'\n')
                        else:
                            train_splits.write(folder+'/'+subfile+'\n')
                        shutil.move(os.path.join(video_dir,folder,file,subfile),os.path.join(video_dir,folder,subfile))
                        i+=1
                    os.rmdir(os.path.join(video_dir,folder,file))
                classInd.write(str(classnum)+' '+folder+'\n')
                classnum+=1
            
            
        
                
