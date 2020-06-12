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

import numpy as np


def main():

    # sys.stdout = open("./log_test_mynet.txt", 'w')
    modelname = "pointcnn_cls"
    #　载入pointcnn
    model = importlib.import_module(modelname)
    # 载入超参数
    setting_path = os.path.join(os.path.dirname(__file__), modelname)
    sys.path.append(setting_path)
    setting = importlib.import_module("mynet")
    sample_num = setting.sample_num 
    rotation_range_val = setting.rotation_range_val
    scaling_range_val = setting.scaling_range_val
    jitter_val = setting.jitter_val
    #　输入文件
    filepath = "../data/mynet/test/1/2018-05-12-12-52-11_Velodyne-HDL-32-Data(1955to2295)_1955.pcd"
    
    batch_size_val = 1 # 不知道干嘛用的参数
    data_frame = np.zeros((batch_size_val, sample_num, 3), dtype=np.float32)
    #frame_id = (filepath.split('_')[-1]).split('.')[0]

    #######################################################################
    # Loading PCD
    with open(filepath, 'r') as f:
            xyz = np.array([ [float(value) for value in line.split(' ')[0:3]]  
                                            for line in f.readlines()[11:len(f.readlines())-1]])
    #######################################################################

    xyz = xyz.astype(np.float32)
    np.random.shuffle(xyz)
    pt_num = xyz.shape[0]
    indices = np.random.choice(pt_num, sample_num, replace=(pt_num <sample_num))
    data_frame[0, ...] = xyz[indices]
    #data_frame[0, ...] = xyz

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
                                                                order=setting.rotation_order)
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

if __name__ == '__main__':
   main()
