#!/usr/bin/env python
# Modified by FCAV, UMich to feedforward all images in a folder
# TODO: Make mean pixels are arguments

import numpy as np
# %matplotlib inline
import Image
import ImageDraw
import argparse
import os
import sys
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from _ensemble import *

# Make sure that the work directory is caffe_root
caffe_root = os.path.join('/root', 'rrc_detection')
os.chdir(caffe_root)
sys.path.insert(0, 'python')


def parse_args():
    parser = argparse.ArgumentParser(description='Feedforward network')
    parser.add_argument('--dataset', dest='dataset',
                        help='path to dataset folder',
                        default='./dataset/', type=str)
    parser.add_argument('--image-list', dest='image_list',
                        help='text file with path to images relative to '
                             'dataset folder',
                        type=str)
    # parser.add_argument('--output-name', dest='output_name',
    #                    help='filename for the output text file', type=str)
    parser.add_argument('--output-dir', dest='output_dir',
                        help='path to output folder',
                        default='/root/rrc_detection/output', type=str)
    parser.add_argument('--iter', dest='iter',
                        help='iter of pretrained model',
                        default='60000', type=str)
    parser.add_argument('--image-height', dest='image_height',
                        help='image height', default='512', type=int)
    parser.add_argument('--image-width', dest='image_width',
                        help='image width', default='1024', type=int)
    parser.add_argument('--model-path', dest='model_path',
                        help='path to model folder',
                        default='models/VGGNet/KITTI/RRC_2560x768_kitti_4r4b_max_size',
                        type=str)
    parser.add_argument('--model-name', dest='model_name',
                        help='name of caffemodel file '
                             'without _iter_<iter number>.caffemodel suffix',
                        default='VGG_KITTI_RRC_2560x768_kitti_4r4b_max_size',
                        type=str)
    parser.add_argument('--gpus', dest='gpu_id',
                        help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--draw-results', dest='draw',
                        help='use ImageDraw to draw bboxes over the image '
                        'and write to the output dir',
                        default=0, type=int)
    args = parser.parse_args()
    return args


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found
    return labelnames


def main(args):
    caffe.set_device(int(args.gpu_id))
    caffe.set_mode_gpu()
    model_def = os.path.join(caffe_root, args.model_path, 'deploy.prototxt')
    model_weights = os.path.join(caffe_root, args.model_path,
                                 args.model_name + '_iter_' + args.iter + '.caffemodel')
    voc_labelmap_file = os.path.join(caffe_root, 'data',
                                     'KITTI-car', 'labelmap_voc.prototxt')
    # save_dir = os.path.join(caffe_root, args.output_dir)
    # txt_dir = os.path.join(caffe_root, args.output_dir)

    detection_out_num = 3
    # if not(os.path.exists(txt_dir)):
    #    os.makedirs(txt_dir)
    # if not(os.path.exists(save_dir)):
    #    os.makedirs(save_dir)
    file = open(voc_labelmap_file, 'r')
    voc_labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), voc_labelmap)
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)   # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 1
    # image_width = args.image_width
    # image_height = args.image_height
    image_width = 2560
    image_height = 768

    net.blobs['data'].reshape(1, 3, image_height, image_width)

    # read images
    with open(args.image_list, 'r') as f:
        lines = f.readlines()
        image_list = [line.strip() for line in lines]
    num_img = len(image_list)

    # result_file = open(os.path.join(save_dir,args.output_name),'w')
    # if not os.path.exits(args.output_dir):
    #    os.makedirs(args.output_dir)

    for img_idx in range(0, num_img):
        file_name = os.path.basename(image_list[img_idx]).replace('.png',
                                                                  '.txt')
        result_file = open(os.path.join(args.output_dir, 'results_txt',
                                        file_name), 'wb')

        det_total = np.zeros([0, 6], float)
        ensemble_num = 0
        img_file = os.path.join(args.dataset, image_list[img_idx])
        print 'processing image {}\n'.format(img_file)
        image = caffe.io.load_image(img_file)

        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # t1 = timeit.Timer("net.forward()","from __main__ import net")
        # print t1.timeit(2)
        # Forward pass.
        net_out = net.forward()
        for out_i in range(2, detection_out_num + 1):
            detections = net_out['detection_out%d' % (out_i)].copy()

            # Parse the outputs.
            det_label = detections[0, 0, :, 1]
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3]
            det_ymin = detections[0, 0, :, 4]
            det_xmax = detections[0, 0, :, 5]
            det_ymax = detections[0, 0, :, 6]
            # Get detections with confidence higher than 0.001
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.001]
            top_conf = det_conf[top_indices]
            # top_label_indices = det_label[top_indices].tolist()
            # top_labels = get_labelname(voc_labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices] * image.shape[1]
            top_ymin = det_ymin[top_indices] * image.shape[0]
            top_xmax = det_xmax[top_indices] * image.shape[1]
            top_ymax = det_ymax[top_indices] * image.shape[0]

            det_this = np.concatenate((top_xmin.reshape(-1, 1),
                                       top_ymin.reshape(-1, 1),
                                       top_xmax.reshape(-1, 1),
                                       top_ymax.reshape(-1, 1),
                                       top_conf.reshape(-1, 1),
                                       det_label[top_indices].reshape(-1, 1)),
                                      1)

            ensemble_num = ensemble_num + 1
            det_total = np.concatenate((det_total, det_this), 0)

        #   evaluate the flipped image
        image_flip = image[:, ::-1, :]
        transformed_image = transformer.preprocess('data', image_flip)
        net.blobs['data'].data[...] = transformed_image
        net_out = net.forward()
        for out_i in range(2, detection_out_num + 1):
            detections = net_out['detection_out%d' % (out_i)].copy()
            temp = detections[0, 0, :, 3].copy()
            detections[0, 0, :, 3] = 1 - detections[0, 0, :, 5]
            detections[0, 0, :, 5] = 1 - temp

            # Parse the outputs.
            det_label = detections[0, 0, :, 1]
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3]
            det_ymin = detections[0, 0, :, 4]
            det_xmax = detections[0, 0, :, 5]
            det_ymax = detections[0, 0, :, 6]

            # Get detections with confidence higher than 0.1.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

            top_conf = det_conf[top_indices]
            # top_label_indices = det_label[top_indices].tolist()
            # top_labels = get_labelname(voc_labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices] * image.shape[1]
            top_ymin = det_ymin[top_indices] * image.shape[0]
            top_xmax = det_xmax[top_indices] * image.shape[1]
            top_ymax = det_ymax[top_indices] * image.shape[0]

            det_this = np.concatenate((top_xmin.reshape(-1, 1),
                                       top_ymin.reshape(-1, 1),
                                       top_xmax.reshape(-1, 1),
                                       top_ymax.reshape(-1, 1),
                                       top_conf.reshape(-1, 1),
                                       det_label[top_indices].reshape(-1, 1)),
                                      1)
            ensemble_num = ensemble_num + 1
            det_total = np.concatenate((det_total, det_this), 0)

        # ensemble different outputs
        det_results = det_ensemble(det_total, ensemble_num)
        idxs = np.where(det_results[:, 4] > 0.0001)[0]
        top_xmin = det_results[idxs, 0]
        top_ymin = det_results[idxs, 1]
        top_xmax = det_results[idxs, 2]
        top_ymax = det_results[idxs, 3]
        top_conf = det_results[idxs, 4]
        # top_label = det_results[idxs, 5]
        if args.draw == 1:
            img = Image.open(img_file)
            draw = ImageDraw.Draw(img)
        for i in xrange(top_conf.shape[0]):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]
            h = float(ymax - ymin)
            w = float(xmax - xmin)
            if (w == 0) or (h == 0):
               continue
            # if (h/w >=2)and((xmin<10)or(xmax > 1230)):
            #    continue
            score = top_conf[i]
            label = 'Car'
            if args.draw == 1:
                if score > 0.5:
                    draw.line(((xmin, ymin), (xmin, ymax), (xmax, ymax),
                               (xmax, ymin), (xmin, ymin)), fill=(0, 255, 0))
                    draw.text((xmin, ymin), '%.2f' % (score),
                              fill=(255, 255, 255))
                elif score > 0.2:
                    draw.line(((xmin, ymin), (xmin, ymax), (xmax, ymax),
                               (xmax, ymin), (xmin, ymin)), fill=(255, 0, 255))
                    draw.text((xmin, ymin), '%.2f' % (score),
                              fill=(255, 255, 255))
            # result_file.write("%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f\n" %
            #                (img_idx, -1, label, -1, -1, -1, xmin, ymin, xmax, ymax, -1, -1, -1, -1, -1, -1, -1, score))
            result_file.write(
             "%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"
             % (label, xmin, ymin, xmax, ymax, score))

        if args.draw == 1:
            # img.save(os.path.join(save_dir,"%06d.png" % (img_idx)))
            img.save(os.path.join(args.output_dir, 'results_image',
                                  os.path.basename(image_list[img_idx])))

        result_file.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
