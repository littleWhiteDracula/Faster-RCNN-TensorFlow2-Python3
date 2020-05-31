# !/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
输出PR曲线, 计算AP值
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lib.nets.vgg16 import vgg16
from lib.datasets.factory import get_imdb
from lib.utils.test import test_net

# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
# 训练输出模型
# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_5000.ckpt',)}
# DATASETS = {'pascal_voc': ('voc_2007_trainval',),
#             'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
NETS = {'vgg16': ('vgg16.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN test')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])  # 模型路径
    # 获得模型文件名称
    filename = (os.path.splitext(tfmodel)[0]).split('\\')[-1]
    filename = 'default' + '/' + filename
    imdb = get_imdb("voc_2007_test")  # 得到
    imdb.competition_mode('competition mode')
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
    # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,  # 记得修改第3个参数为：类别数量+1
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))
    print(filename)
    test_net(sess, net, imdb, filename, max_per_image=100)
    sess.close()
