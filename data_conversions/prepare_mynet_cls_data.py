#!/usr/bin/python3
'''Convert point cloud from velodyne to h5.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import argparse
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    args = parser.parse_args()
    print(args)

    # batch_size = 2048
    batch_size = 64     #几帧点云打包成一个h5文件
    sample_num = 2048  #单帧采样点数

    # 数据文件夹
    folder_mynet = args.folder if args.folder else '../../data' # 类似于c语言的 ? : 表达式
    train_test_folders = ['train', 'test']

    
    data = np.zeros((batch_size, sample_num, 3)) # 3要根据点云类型进行修改,表示通道数
    label = np.zeros((batch_size), dtype=np.int32)

    

    for folder in train_test_folders: # 获取数据文件夹下的train和test子文件夹

        folder_pts = os.path.join(folder_mynet, folder)
        filename_filelist_h5 = os.path.join(folder_mynet, '%s_files.txt' % folder) # 存 文件名列表 的文件
        idx_h5 = 0
        
        for label_id in os.listdir(folder_pts):
            folder_label = os.path.join(folder_pts, label_id) # 获取各标签文件夹路径
            filelist = os.listdir(folder_label) # 所有点云文件名列表
            
            for idx_pts, filename in enumerate(filelist): # enumerate能同时返回idx和元素,因为batch_size需要文件编号
                
                filename_pts = os.path.join(folder_label, filename)  # 点云所在路径
                with open(filename_pts) as f: # 读取pcd文件
                    xyzi_array = np.array([ [float(value) for value in line.split(' ')]
                                                        for line in f.readlines()[11:len(f.readlines())-1]]) # 从12行到末尾
                '''
                np.random.shuffle(xyzi_array) # 打乱点云中的点,为了随机采样 
                pt_num = xyzi_array.shape[0] # 当前帧总点数
                indices = np.random.choice(pt_num, sample_num, replace=(pt_num < sample_num)) # 若总点数少于采样数则进行重复采样
                points_array = xyzi_array[indices] # 取得采样后的点云数据
                #points_array[..., 3:] = points_array[..., 3:]/255 - 0.5 # normalize colors
                '''
                idx_in_batch = idx_pts % batch_size
                #data[idx_in_batch, ...] = points_array
                data[idx_in_batch, ...] = xyzi_array
                label[idx_in_batch] = int(label_id)
                
                # 打包为hdf5
                if ((idx_pts + 1) % batch_size == 0) or idx_pts == len(filelist) - 1:
                    item_num = idx_in_batch + 1
                    
                    filename_h5 = os.path.join(folder_mynet, '%s_%d.h5' % (folder, idx_h5))
                    print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                    
                    with open(filename_filelist_h5, 'a') as filelist_h5: #准备写入训练测试的文件名列表
                        filelist_h5.write('./%s_%d.h5\n' % (folder, idx_h5))

                    file = h5py.File(filename_h5, 'w')
                    file.create_dataset('data', data=data[0:item_num, ...])
                    file.create_dataset('label', data=label[0:item_num, ...])
                    file.close()

                    idx_h5 = idx_h5 + 1

if __name__ == '__main__':
    main()
