#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
将Faster-RCNN-TensorFlow-Python3 训练得ckpt格式转换成PB格式
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
from tensorflow.python.framework import graph_util
# from tensorboard.plugins.graph import graph_util

# from lib.nets.resnet_v1 import resnetv1

tf.disable_v2_behavior()
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer


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

    plt.axis('off')
    plt.tight_layout()
    plt.draw()



def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    # opencv读取图片
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
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

    # cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # f_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    f_path = r"G:\PyCharm\PyCharmSpaceWork\Faster-RCNN-TensorFlow2-Python3"

    demonet = "vgg16"
    dataset = "pascal_voc"
    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    # input_dir = r"D:\result\ship_tf\input_data"
    input_dir = r"G:\PyCharm\PyCharmSpaceWork\Faster-RCNN-TensorFlow2-Python3"
    # output_dir = r"D:\result\ship_tf"
    output_dir = r"G:\PyCharm\PyCharmSpaceWork"
    # tfmodel = os.path.join(f_path, 'output', demonet, DATASETS[dataset][0], 'default_ship',
    #                        NETS[demonet][0])
    tfmodel = os.path.join(f_path, 'output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    print(tfconfig)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    # elif demonet == 'res101':
    #     net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    net.create_architecture(sess,"TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    # ckpt to pb
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph = r"G:\PyCharm\PyCharmSpaceWork\ship_model_from_demo3.pb"
    # output_node_names = "vgg_16_3/cls_prob,add,vgg_16_1/rois/concat,vgg_16_3/cls_score/BiasAdd"
    output_node_names = "vgg_16/cls_score/BiasAdd,vgg_16/cls_prob,add,vgg_16/rois/PyFunc"
    # output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,
    #                                                              output_node_names.split(","))
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                 output_node_names.split(","))
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("PB保存得路径是:\n", output_graph)

    # end running time
    endtime = datetime.datetime.now()
    print("程序运行时间:", endtime - starttime)

