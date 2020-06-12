#!/usr/bin/python3
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import importlib
import numpy as np
import pointfly as pf
import tensorflow as tf
import time


def load_pcd(filepath): 
    with open(filepath, 'r') as f:
            xyz = np.array([ [float(value) for value in line.split(' ')[0:3]]  
                                            for line in f.readlines()[11:len(f.readlines())-1]])
    xyz = xyz.astype(np.float32)
    return xyz

def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    else:
        print("No such file:%s" % filepath)
        return False 

def main():
    modelname = "pointcnn_cls"
    settingname = "mynet_2048"
    load_ckpt = "../models/ckpts/iter-5007"
    temp_dir = "../haimai/tmp" # 存放临时文件的位置
    if(os.path.exists(temp_dir) == False):
        os.makedirs(temp_dir)
        
    temp_frame = os.path.join(temp_dir, "cur_frame.pcd") # 存放当前帧的位置
    temp_isend = os.path.join(temp_dir, "isend")
    temp_c_writing = os.path.join(temp_dir, "c_is_writing")

    
    setting_path = os.path.join(os.path.dirname(__file__), modelname)
    sys.path.append(setting_path)
    setting = importlib.import_module(settingname) ### 载入超参数
    
    batch_size_val = 1 # 一次只预测一帧点云
    data_frame = np.zeros((batch_size_val, setting.sample_num, 3), dtype=np.float32)
    
    ### Placeholders
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    is_training = tf.placeholder(tf.bool, name='is_training')
    data_frame_placeholder = tf.placeholder(data_frame.dtype, data_frame.shape, name="data_frame")    
    
    model = importlib.import_module(modelname) # 载入pointcnn
    points_augmented = pf.augment(data_frame_placeholder, xforms, jitter_range)
    net = model.Net(points=points_augmented, features=None, is_training=is_training, setting=setting) # 构建网络,暂时不使用其他特征
    logits = net.logits
    probs = tf.nn.softmax(logits, name='probs')
    predict = tf.argmax(probs, axis=-1, name='predictions') # 网络的输出
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_ckpt) # Load the model
        xforms_np, rotations_np = pf.get_xforms(batch_size_val, rotation_range=setting.rotation_range_val,
                                                                scaling_range=setting.scaling_range_val,
                                                                order=setting.rotation_order)
        ### 循环进行单帧的标签预测
        while (os.path.exists(temp_isend) == False): # 非最后一帧
            if (os.path.exists(temp_frame) == True): # 点云帧
                while (os.path.exists(temp_c_writing) == True): # C++ is writing PCD
                    time.sleep(0.05) 
                data_frame[0, ...] = load_pcd(temp_frame) # 读取点云
                label = sess.run(predict, feed_dict={data_frame_placeholder: data_frame,
                                                            xforms: xforms_np,
                                                            rotations: rotations_np,
                                                            jitter_range: np.array([setting.jitter_val]),
                                                            is_training: False,
                                                            })
                ### 传递预测结果给mapping
                label_file = open(temp_dir + "/label_" + str(label[0][0]), 'w')
                label_file.close()
                
                if delete_file(temp_frame) == False:
                    print("Delete frame error.")
                    return     
    delete_file(temp_isend) # 删除通信文件
    return 

if __name__ == '__main__':
   main()
