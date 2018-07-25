from pycocotools.coco import COCO
from numpy import *
import os

data_dir = '.'
data_type = 'val2014'
output_caronly_dir = '%s/annotations_txt' % (data_dir)

if not os.path.exists(output_caronly_dir):
    os.makedirs(output_caronly_dir)

# initialize COCO api for instance annotations
coco = COCO('%s/annotations/instances_%s.json' % (data_dir,data_type))

imgs = coco.loadImgs(list(coco.imgs.keys()))
for k, img in enumerate(imgs):
    print('[%6d/%6d] %s' % (k, len(imgs), img['file_name']))
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))

    image_name, image_ext= os.path.splitext(img['file_name'])
    result_file_car_only = open(
        os.path.join(output_caronly_dir,
                    img['file_name'].replace(image_ext, '.txt')),
        'w')

    for ann in anns:
        category = ann['category_id']

        if category == 3:
            bbox = ann['bbox']
            label = 'Car'
            result_file_car_only.write(
                "%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10\n"
                % (label, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
