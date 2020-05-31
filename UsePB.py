from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import datetime
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.utils.filePath import filePath
from lib.utils.test import _get_blobs
from lib.utils.test import _clip_boxes
from lib.utils.bbox_transform import bbox_transform_inv

CLASSES = ('__background__', 'roses')


NETS = {'vgg16': ('vgg16.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval',)}

def vis_detections(im,class_name, dets,outfile, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        #print(bbox)
        score = dets[i, -1]

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),0]=0
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),0] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),0] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),0] = 0

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),1]=0
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),1] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),1] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),1] = 0

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),2]=255
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),2] = 255
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),2] = 255
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),2] = 255

def demo(image_name,out_file,sess):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im=cv2.imread(image_name)

    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    print(blobs["im_info"])

    #Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    print("====freeze_graph_test()-> blobs====\n", blobs)
    scores, boxes = freeze_graph_test(sess, blobs)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    #

    im = im[:, :, (2, 1, 0)]

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im,cls, dets,out_file, thresh=CONF_THRESH)

    cv2.imencode('.jpg',im)[1].tofile(out_file)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')

    parser.add_argument('--input',dest='input',help='input folder')
    parser.add_argument('--output',dest='output',help='output folder')
    args = parser.parse_args()

    return args

def freeze_graph_test(sess, blobs):
    '''
	:param pb_path:pb文件的路径
	:param image_path:测试图片的路径
	:return:
	'''
    # 定义输入的张量名称,对应网络结构的输入张量
    # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
    # 定义输出的张量名称
    input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
    tensor_info = sess.graph.get_tensor_by_name("Placeholder_1:0")

    biasadd = sess.graph.get_tensor_by_name("vgg_16/cls_score/BiasAdd:0")
    score = sess.graph.get_tensor_by_name("vgg_16/cls_prob:0")
    bbox = sess.graph.get_tensor_by_name("add:0")
    rois = sess.graph.get_tensor_by_name("vgg_16/rois/PyFunc:0")

    # input_image_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    # tensor_info = tf.placeholder(tf.float32, shape=[3, ])
    # print("输入tensor")
    # print(input_image_tensor)
    # print(tensor_info)
    feed_dict = {
        input_image_tensor: blobs['data'],
        tensor_info: blobs['im_info']
    }
    # 模型预测结果 返回四个值
    # scores、bbox_pred、rois，分别是检测框分数，修正值，检测框
    _, scores, bbox_pred, rois = sess.run([biasadd, score, bbox, rois],
                                          feed_dict=feed_dict)
    print("=======blobs==========\n", blobs)
    im_scales=blobs['im_info'][2]
    # 检测框修正
    boxes = rois[:, 1:5] / im_scales
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, (255,255,0))

    return scores, pred_boxes

if __name__ == '__main__':

    input_dir = r"G:\PyCharm\PyCharmSpaceWork\Faster-RCNN-TensorFlow2-Python3\data\demo"
    output_dir = r"‪G:\PyCharm\PyCharmSpaceWork"
    pb_path = r"G:\PyCharm\PyCharmSpaceWork\ship_model_from_demo3.pb"
    im_names = glob.glob(os.path.join(input_dir,"*.jpg"))
    print(im_names)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for im_name in im_names:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                img_basename = os.path.basename(im_name)
                out_file = os.path.join(output_dir,img_basename)
                print("========out_file============\n", out_file)
                print(im_name)
                demo(im_name,out_file,sess)
