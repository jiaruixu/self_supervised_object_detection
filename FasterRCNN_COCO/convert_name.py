import os
import glob

_FOLDERS_MAP = {
    'prediction': './test_txt',
    'gt': './annotations_txt',
}

_OUTPUT_MAP = {
    'prediction': './test_txt_rename',
    'gt': './annotation_txt_rename',
}

def _get_files(data):
    pattern = '*.txt'
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main():
    prediction_files = _get_files('prediction') # prediction
    gt_files = _get_files('gt') # ground truth
    num_files = len(gt_files )

    for i in range(num_files):
        pre_name, _ = os.path.splitext(os.path.basename(prediction_files[i]))
        gt_name, _ = os.path.splitext(os.path.basename(gt_files[i]))
        print(">>precessing image %s" % (gt_name))
        if gt_name != pre_name:
            print(">>ground truth %s is not correspond to %s" % (gt_name, pre_name))
            break

        new_pattern = '%06d.txt' % (i)

        if not os.path.exists(_OUTPUT_MAP['gt']):
            os.mkdir(_OUTPUT_MAP['gt'])
        gt_new_file = os.path.join(_OUTPUT_MAP['gt'], new_pattern)
        gt = open(gt_files[i])
        gt_new = open(gt_new_file, "w")
        for line in gt.readlines():
            gt_new.write(line)

        if not os.path.exists(_OUTPUT_MAP['prediction']):
            os.mkdir(_OUTPUT_MAP['prediction'])
        pre_new_file = os.path.join(_OUTPUT_MAP['prediction'], new_pattern)
        pre = open(prediction_files[i])
        pre_new = open(pre_new_file, "w")
        for line in pre.readlines():
            pre_new.write(line)

if __name__ == '__main__':
    main()
