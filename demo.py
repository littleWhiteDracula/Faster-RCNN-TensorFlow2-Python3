#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.utils.filePath import filePath

# 计算mAP的值
from lib.utils.test import test_net
from lib.datasets.factory import get_imdb

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'roses')

# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
#         'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
NETS = {'vgg16': ('vgg16.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval',)}
# 将检测完的数据保存起来
SAVA_DIR = 'output/sign'


def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes.
        image_name:图片的名字
    """
    # [:, -1] 取每个元素列表的最后一个值
    # np.where() 满足条件(condition 为True or False)，输出x，不满足输出y
    # 当没有指定返回的值时, array的值是列表中满足条件的所有值
    # eg:[0,1,2,...,9,10] 条件>5 return array([6,7,..,9])
    # 返回的是一个array
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    # 创建子图 figsize=(12, 12)子图的大小
    # 返回一个figure图像和子图ax的array列表
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    # inds [0]
    # dets [[ 0.         7.872568  63.        55.951324   0.7933805]]
    # i 0
    # score 0.7933805
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print("========inds========\n", inds)
        # print("=====dets======\n", dets)
        # print("=========i=========\n", i)
        # print("========score=========\n", score)

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    # ax子图设置标题
    # {:.1f} 位置映射 取n位小数
    ax.set_title(('{}=={} detections with '
                  'p({} | box) >= {:.1f}').format(image_name, class_name,
                                                  class_name,
                                                  thresh),
                 fontsize=14)
    # 将验证结果传入到列表里
    verificationResult.append(image_name)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()



def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # G:\PyCharm\PyCharmSpaceWork\Faster-RCNN-TensorFlow2-Python3\data\demo\000001.jpg
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    # print("==================im_file===========", im_file)
    # opencv读取图片
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    # 调用了lib/model/test.py里的im_detect()方法，返回的是分数和检测框。
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        '''cls 标签的类型 roses'''
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        # [[ 0.         7.872568  63.        55.951324   0.7933805]]
        # 最后一个值是相似度
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name=image_name, thresh=CONF_THRESH )

    # 保存标记的图片
    if not os.path.exists(SAVA_DIR):
        os.makedirs(SAVA_DIR)
    plt.savefig(os.path.join(SAVA_DIR, image_name))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    # parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
    #                     choices=NETS.keys(), default='res101')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # start running time
    starttime = datetime.datetime.now()

    # 新建一个变量, 用来存储验证集中都有哪些被验证了
    verificationResult = []

    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])


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

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg',
    #             '000005.jpg', '000006.jpg', '000007.jpg', '000008.jpg',
    #             '000009.jpg', '000010.jpg', '000011.jpg', '000012.jpg',
    #             '000013.jpg', '000014.jpg', '000015.jpg', '000016.jpg',
    #             '000017.jpg', '000018.jpg', '000019.jpg', '000020.jpg',
    #             '000021.jpg', '000022.jpg', '000023.jpg', '000024.jpg',
    #             '000025.jpg', '000026.jpg', '000027.jpg',]
    im_names = ['000001.jpg']
    # im_names = []
    # filePath = filePath()
    # root, dirs, files = filePath.file_name(r"G:\PyCharm\PyCharmSpaceWork\Faster-RCNN-TensorFlow2-Python3\data\demo")
    # im_names = files
    # print("====================\n", im_names)

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    # imdb = get_imdb("voc_2007_trainval")
    # test_net(sess, net, imdb, 'default')
    # end running time
    endtime = datetime.datetime.now()
    # 输出验证结果集
    print("========验证结果集========\n",verificationResult)

    print("程序运行时间:", endtime - starttime)
    plt.show()
