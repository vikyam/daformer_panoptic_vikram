import os
from PIL import Image
import numpy as np
import imageio
from map_ids import get_map
import pickle
from map_ids import get_map_s2c_19cls
import argparse

'''
commands to execute:
run these all in parallel

$ cdcvpr2022
$ screen
$ source ~/daformer/bin/activate
$ PYTHONPATH="/home/suman/apps/code/CVPR2022/daformer_panoptic:$PYTHONPATH" && export PYTHONPATH
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 0 --max 1000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 1000 --max 2000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 2000 --max 3000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 3000 --max 4000  
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 4000 --max 5000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 5000 --max 6000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 6000 --max 7000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 7000 --max 8000
$ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_pickle_files_19cls.py --min 8000 --max 10000
'''
parser = argparse.ArgumentParser(description='pickle file generation')
parser.add_argument("--min", type=int, default=0, help='min')
parser.add_argument("--max", type=int, default=1000, help='max')
args = parser.parse_args()
print('args.min:{}   args.max: {}'.format(args.min, args.max))
DEBUG = False
VIS = False # False  # visualization
_SYNTHIA_THING_LIST = [10, 17, 8, 18, 19, 20, 12, 11]
crowd_th_list = [0] # [0, 1000, 2000]

for crowd_region_threshold in crowd_th_list:
    imgid = 0
    label_path = '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES/GT/LABELS'
    file_list = os.listdir(label_path)
    file_list.sort()

    if not DEBUG:
        out_path1 = '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES/GT/' \
                    'panoptic-labels-pklfiles-crowdth-{}-for-daformer'.format(crowd_region_threshold)
    else:
        out_path1 = '/home/suman/Downloads/work_march_23_2022/synthia_gt_visual/pickle_files-crowdth-{}'.format(crowd_region_threshold) # DEBUG time use
    if not os.path.exists(out_path1):
        os.makedirs(out_path1)

    map_syn_2_city = get_map_s2c_19cls()

    for f in file_list:
        if imgid >= args.min and imgid < args.max:
            file_name = os.path.basename(f)
            f = os.path.join(label_path, f)
            raw_label = np.asarray(imageio.imread(f, format='PNG-FI'))
            semantic_labels = np.uint8(raw_label[:, :, 0])
            semantic_labels_uids = np.unique(semantic_labels)
            instance_labels = np.uint16(raw_label[:, :, 1])
            instance_labels_uids = np.unique(instance_labels)

            if not VIS:
                instance_labels_new = np.zeros((instance_labels.shape[0], instance_labels.shape[1]), dtype=np.uint32)
                instance_labels_new_outfile = os.path.join(out_path1, file_name.replace('png', 'pkl'))
            else:
                instance_labels_new = np.zeros((instance_labels.shape[0], instance_labels.shape[1]), dtype=np.uint8)
                instance_labels_new_outfile = os.path.join(out_path1, file_name)

            # loop over semanitc classes
            for sl in semantic_labels_uids:
                mask_sem = semantic_labels == sl  # get the mask for the segment (set of pixels) where semantic label = sl
                segment_count = 0
                # for synthia class id sl get the corresponding cityscapes class id
                NOT_IN_19_CLS_SET = False
                try:
                    mapped_sl = map_syn_2_city[sl]  # converting synthia class id to cityscapes class id # refer to NOTE-1
                except KeyError:
                    mapped_sl = 0 # refer to NOTE-2 written above
                    NOT_IN_19_CLS_SET = True
                # for each semanitc class find out the corresponding segments
                for il in instance_labels_uids:
                    mask_ins = instance_labels == il  # get the mask for the segment (set of pixels) where instance label = il
                    mask = np.logical_and(mask_ins, mask_sem)  # get the intersection region of the instance and segment mask
                    if mask.any():
                        if NOT_IN_19_CLS_SET:
                            new_label_id = mapped_sl # mapped_sl = 0 # pixels belong to any class which are not in the common 16 cls are set to 0
                        else:
                            area = np.sum(mask)  # segment area computation
                            # set the isCrowd flag - 0: no crowd, 1: crowd region
                            if area > crowd_region_threshold:
                                isCrowd = 0
                            else:
                                isCrowd = 1
                            # based on the isCrowd flag, generate the panoptic id for this segment
                            if isCrowd == 0: # if not in crowd region,
                                if sl in _SYNTHIA_THING_LIST: # thing ids are >= 1000
                                    new_label_id = mapped_sl * 1000 + segment_count
                                    segment_count += 1
                                else:
                                    new_label_id = mapped_sl # stuff ids are < 1000
                            # if segment in crowd region (or the class is not in 16 common cls set) then both stuff and thing class ids are < 1000
                            elif isCrowd == 1:
                                new_label_id = mapped_sl
                        instance_labels_new[mask] = new_label_id

            if VIS:
                imageio.imwrite(instance_labels_new_outfile, instance_labels_new, format='PNG-FI')
                print('done file: {}'.format(instance_labels_new_outfile))

            if not VIS:
                with open(instance_labels_new_outfile, 'wb') as f:
                    pickle.dump(instance_labels_new, f)
                    print('done file: {}'.format(instance_labels_new_outfile))

        imgid+=1