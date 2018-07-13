## Docker image

Build the image
```
cd ngv-dockerfiles/
docker build -t caffe-rrc ./caffe-rrc
```

Use the image

```
nvidia-docker run --rm -it caffe-rrc
```

## Use container function
```
container-fn caffe-rrc-test --modeldir-path /mnt/ngv/self-supervised-learning/models_rrc \
                            --dataset-path /mnt/ngv/datasets/cityscapes \
                            --image-list ./dataset/data/cityscapes_image_list.txt \
                            --output-path /mnt/fcav/self_training/object_detection/rrc_cityscapes/results \
                            --draw-results 1 \
                            --gpus 0
```

## To do
`caffe-rrc-train` has not been finished yet.
