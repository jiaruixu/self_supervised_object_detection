## Set environment variable

```
# From container functions repo
$ export CONTAINER_FN_PATH=`pwd`/kitti-tracking-fns
```

Or add this to `.bashrc` to make it stick, e.g. (assume you also want to add path to `container-fns`)

```
export CONTAINER_FN_PATH=/path/to/container-fns:/path/to/kitti-tracking-fns
```

## Build image list
The directory structure in the image is
```
+ /root/rrc_detection/
    + rrc_feedforward.py
    + dataset
      + data
    + output
```

As working path is `/root/rrc_detection/`, the dataset path in `rrc_feedforward.py` is set as `./dataset/` and the dataset path has been mounted to `/root/rrc_detection/dataset/data`, the format for the `image_list_left.txt` or `image_list_right.txt` should be

```
./data/{relative_path_to_images}/image_name1.png
./data/{relative_path_to_images}/image_name2.png
./data/{relative_path_to_images}/image_name3.png
...
```

## Use container-fn

```
container-fn caffe-rrc-test --modeldir-path /mnt/ngv/self-supervised-learning/models_rrc \
                            --dataset-path /mnt/ngv/datasets/cityscapes \
                            --image-list ./dataset/data/cityscapes_image_list.txt \
                            --output-path /mnt/fcav/self_training/object_detection/rrc_cityscapes/results \
                            --draw-results 1 \
                            --gpus 0
```

output:

```
mkdir -p /mnt/ngv/self-supervised-learning/Datasets/KITTI/tracking/training/det_rrc
(dry run, command not run)


nvidia-docker run --rm \
  --entrypoint '' \
  -v /mnt/ngv/datasets/cityscapes/training/image_2:/root/rrc_detection/dataset/data:ro \
  -v /mnt/fcav/self_training/object_detection/rrc_cityscapes/results:/root/rrc_detection/output \
  -v /mnt/ngv/self-supervised-learning/models_rrc:/root/rrc_detection/models:ro \
  -v /home/jiarui/git/kitti-tracking-fns/fns/caffe-rrc-test/static/root/rrc_detection/rrc_feedforward.py:/root/rrc_detection/rrc_feedforward.py:ro \
  -it caffe-rrc \
  python rrc_feedforward.py \--
  --image-list /home/jiarui/git/self_supervised_object_detection/cityscapes_image_list.txt \
  --iter 60000 \
  --model-path models/VGGNet/KITTI/RRC_2560x768_kitti_4r4b_max_size \
  --model-name VGG_KITTI_RRC_2560x768_kitti_4r4b_max_size \
  --gpus 0 \
  --draw-results 1 \
  2>&1 | tee -a /tmp/testing-logs.txt
(dry run, command not run)
```
