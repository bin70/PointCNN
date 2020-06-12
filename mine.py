#!/usr/bin/python3
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime
import time

print(tf.__path__)
print(tf.__version__)

balance_fn = None
map_fn = None
keep_remainder = True
save_ply_fn = None

num_class = 2
data_dim = 3 # 根据点云类型进行修改

sample_num = 1024 # 采样点数
sampling = 'random'

#batch_size = 128
batch_size = 1

#num_epochs = 1024
num_epochs = 10

step_val = 500 # 这个参数干嘛的不知道

learning_rate_base = 0.01
decay_steps = 8000
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 1e-5

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.1, 0.1, 0.1, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 3

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]

with_global = True

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.8)]]

optimizer = 'adam'
epsilon = 1e-2

use_extra_features = False
with_X_transformation = True
sorting_method = None

def func2(filepath):
    # sys.stdout = open("./log_test_mynet.txt", 'w')
    modelname = "pointcnn_cls"
    # 载入pointcnn
    
    model = importlib.import_module(modelname)
    
    # 载入超参数
    #setting_path = os.path.join(os.path.dirname(__file__), modelname)
    #sys.path.append(setting_path)
    #setting = importlib.import_module("mynet")
    #sample_num = sample_num 
    #rotation_range_val = rotation_range_val
    #scaling_range_val = scaling_range_val
    #jitter_val = jitter_val
    #输入文件
    #filepath = "../data/mynet/test/1/2018-05-12-12-52-11_Velodyne-HDL-32-Data(1955to2295)_1955.pcd"
    
    batch_size_val = 1 # 不知道干嘛用的参数
    data_frame = np.zeros((batch_size_val, sample_num, 3), dtype=np.float32)
    #frame_id = (filepath.split('_')[-1]).split('.')[0]
    
    #######################################################################
    # Loading PCD
    with open("../data/mynet/test/1/2018-05-12-12-52-11_Velodyne-HDL-32-Data(1955to2295)_1955.pcd", 'r') as f:
            xyz = np.array([ [float(value) for value in line.split(' ')[0:3]]  
                                            for line in f.readlines()[11:len(f.readlines())-1]])
    #######################################################################
    '''
    xyz = xyz.astype(np.float32)
    np.random.shuffle(xyz)
    #pt_num = xyz.shape[0]
    #indices = np.random.choice(pt_num, sample_num, replace=(pt_num <sample_num))
    #data_frame[0, ...] = xyz[indices]
    data_frame[0, ...] = xyz
    
    ######################################################################
    # Placeholders
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    is_training = tf.placeholder(tf.bool, name='is_training')
    data_frame_placeholder = tf.placeholder(data_frame.dtype, data_frame.shape, name="data_frame")    
    ######################################################################

    
    #######################################################################
    # 网络的输入
    points_augmented = pf.augment(data_frame_placeholder, xforms, jitter_range)
    # 构建网络,暂时不使用其他特征
    net = model.Net(points=points_augmented, features=None, is_training=is_training, setting=setting)
    logits = net.logits
    probs = tf.nn.softmax(logits, name='probs')
    # 网络的输出
    predict = tf.argmax(probs, axis=-1, name='predictions')
    #######################################################################
    
    #######################################################################
    load_ckpt = "/home/elvin/models/mynet/pointcnn_cls_mynet_2019-07-05-20-00-17_31693/ckpts/iter-528"
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_ckpt) # Load the model
        xforms_np, rotations_np = pf.get_xforms(batch_size_val, rotation_range=rotation_range_val,
                                                                scaling_range=scaling_range_val,
                                                                order=rotation_order)
        res = sess.run(predict, feed_dict={data_frame_placeholder: data_frame,
                                                            xforms: xforms_np,
                                                            rotations: rotations_np,
                                                            jitter_range: np.array([jitter_val]),
                                                            is_training: False,
                                                            })
        print("res=", res[0][0])    
    ######################################################################    
    #sys.stdout.flush()
    return res[0][0]
    '''
    return 0

