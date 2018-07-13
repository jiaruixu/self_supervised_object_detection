from __future__ import print_function

import math
import os
import shutil
import stat
import subprocess
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
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--train-data', dest='train_data',
                        help='The database file for training data.',
                        default='./dataset/', type=str)
    parser.add_argument('--model-name', dest='model_name', default='kitti_car',
                        type=str)
    parser.add_argument('--branch-num', dest='branch_num', default=4,
                        type=int)
    parser.add_argument('--rolling-time', dest='rolling_time', default=4,
                        type=int)
    parser.add_argument('--run-soon', dest='run_soon',
                        help='Set true if you want to start training right'
                        ' after generating all files.', action='store_true')
    parser.add_argument('--resume-training', dest='resume_training ',
                        help='Set true if you want to load from most recently'
                        ' saved snapshot. Otherwise, we will load from the'
                        ' pretrain_model defined below.', action='store_true')
    parser.add_argument('--remove-old-models', dest='remove_old_models',
                        help='If true, Remove old model files.',
                        action='store_false')
    parser.add_argument('--resize-height', dest='resize_height ',
                        help='image height', default=768, type=int)
    parser.add_argument('--resize-width', dest='resize_width',
                        help='image width', default=2560, type=int)
    parser.add_argument('--learning-rate', dest='learning_rate',
                        default=0.0005, type=int)
    parser.add_argument('--use-batchnorm', dest='use_batchnorm',
                        action='store_false')
    parser.add_argument('--name-size-file', dest='name_size_file',
                        default='data/KITTI-car/testing_name_size.txt',
                        help='Stores the test image names and sizes.',
                        type=str)
    parser.add_argument('--model-path', dest='model_path',
                        help='path to model folder',
                        default='models/VGGNet/KITTI/RRC_2560x768_kitti_4r4b_max_size',
                        type=str)
    parser.add_argument('--pretrain-model', dest='pretrain_model',
                        help='The pretrained model. We use the Fully'
                        ' convolutional reduced (atrous) VGGNet.',
                        default='models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel',
                        type=str)
    parser.add_argument('--label-map-file', dest='label_map_file',
                        help='Stoeres LabelMapItem.',
                        default='data/KITTI-car/labelmap_voc.prototxt',
                        type=str)
    parser.add_argument('--gpus', dest='gpulist',
                        help='GPU devices to evaluate with', nargs='*',
                        default=[0, 1], type=str)
    args = parser.parse_args()
    return args


def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
    ConvBNLayer(net, "conv4_3", "conv4_3r", use_batchnorm, use_relu, 256, 3, 1, 1)
    ConvBNLayer(net, "fc7", "fc7r", use_batchnorm, use_relu, 256, 3, 1, 1)

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    for i in xrange(7, 9):
        from_layer = out_layer
        out_layer = "conv{}_1".format(i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1)

        from_layer = out_layer
        out_layer = "conv{}_2".format(i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    return net


def main(args):
    rolling_time = 4
    branch_num = 4
    model_name = "kitti_car"
    # Set true if you want to start training right after generating all files.
    run_soon = True
    # Set true if you want to load from most recently saved snapshot.
    # Otherwise, we will load from the pretrain_model defined below.
    resume_training = True
    # If true, Remove old model files.
    remove_old_models = False

    # The database file for training data. Created by data/KITTI/create_data.sh
    train_data = "data/KITTI-car/lmdb/KITTI-car_training_lmdb"
    # The database file for testing data. Created by data/KITTI/create_data.sh
    test_data = "data/KITTI-car/lmdb/KITTI-car_testing_lmdb"
    # Specify the batch sampler.

    resize_width = 2560
    resize_height = 768
    resize = "{}x{}".format(resize_width, resize_height)

    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    learning_rate = 0.0005
    use_batchnorm = False
    # Use different initial learning rate.
    if use_batchnorm:
        base_lr = 0.04
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = 0.00004
    # Modify the job name if you want.
    job_name = "RRC_{}_{}".format(resize,model_name)
    # The name of the model. Modify it if you want.
    model_name = "VGG_KITTI_{}".format(job_name)

    # Directory which stores the model .prototxt file.
    save_dir = "models/VGGNet/KITTI/{}".format(job_name)
    # Directory which stores the snapshot of models.
    snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
    # Directory which stores the job script and log file.
    job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
    # model definition files.
    train_net_file = "{}/train.prototxt".format(save_dir)
    test_net_file = "{}/test.prototxt".format(save_dir)
    deploy_net_file = "{}/deploy.prototxt".format(save_dir)
    solver_file = "{}/solver.prototxt".format(save_dir)
    # snapshot prefix.
    snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
    # job script path.
    job_file = "{}/{}.sh".format(job_dir, model_name)

    # Stores the test image names and sizes. Created by data/KITTI/create_list.sh
    name_size_file = "data/KITTI-car/testing_name_size.txt"
    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
    # Stores LabelMapItem.
    label_map_file = "data/KITTI-car/labelmap_voc.prototxt"
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1]
    normalizations2 = [-1, -1, -1, -1, -1]
    num_outputs=[256,256,256,256,256]
    odd=[0,0,0,0,0]
    rolling_rate = 0.075
    # Solver parameters.
    # Defining which GPUs to use.
    gpus = "0,1,2,3"
    gpulist = gpus.split(",")
    num_gpus = len(gpulist)

    # Divide the mini-batch to different GPUs.
    batch_size = 4
    accum_batch_size = 8
    iter_size = accum_batch_size / batch_size
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = batch_size
    if num_gpus > 0:
      batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
      iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
      solver_mode = P.Solver.GPU
      device_id = int(gpulist[0])



if __name__ == '__main__':
    args = parse_args()
    main(args)
