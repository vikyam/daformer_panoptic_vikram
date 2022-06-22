
# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image
import imageio

# cityscapes imports
# from cityscapesscripts.helpers.csHelpers import printError
# from cityscapesscripts.helpers.labels import id2label, labels
# from cityscapes_labels_16cls import id2label
import pickle
# for drawing bbox and visualize for debug
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from map_ids import get_map
from cityscapesscripts.helpers.labels import id2label, labels

'''
commands to execute:
    $ cdcvpr2022
    $ screen
    $ source ~/daformer/bin/activate
    $ PYTHONPATH="/home/suman/apps/code/CVPR2022/daformer_panoptic:$PYTHONPATH" && export PYTHONPATH
    $ python synthiascripts/save_panopitc_gt_labels_for_synthia_as_color_png_file_19cls.py

'''

# try:
#     categoryId = id2trainId[semanticId]
# except KeyError:
#     categoryId = 255
# if categoryId == 255:
#     continue
# if semanticId not in _SYNTHIA_THING_LIST:
#     isCrowd = 0

# The main method
def convert2panoptic(synthiaPath=None, outputFolder=None):

    # id2trainId = get_map()

    # if you want to visualize the boxes around each segments for debug purpose
    BBOX_VIS = False

    file_list = os.listdir(synthiaPath)
    file_list.sort()
    # a bit verbose
    print("Converting {} annotation files".format(len(file_list)))
    outputBaseFile = "synthia_panoptic"
    outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
    print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
    panopticFolder = os.path.join(outputFolder, outputBaseFile)
    if not os.path.isdir(panopticFolder):
        print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
        os.mkdir(panopticFolder)
    print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))
    images = []
    annotations = []
    useTrainId = True
    for progress, f in enumerate(file_list):

        if BBOX_VIS:
            lineWidth = 1
            cc = 'r'
            fig, ax = plt.subplots(1, 2)
            bbox_vis = []

        f = os.path.join(synthiaPath, f)
        # originalFormat = np.array(Image.open(f))
        # originalFormat = np.asarray(imageio.imread(f, format='PNG-FI'))
        with open(f,'rb') as f2o:
            originalFormat = pickle.load(f2o)
        fileName = os.path.basename(f)
        imageId = fileName.replace(".pkl", "")
        inputFileName = fileName.replace(".pkl", "_panoptic.png")
        outputFileName = inputFileName
        # image entry, id for image is its filename without extension
        images.append({"id": imageId, "width": int(originalFormat.shape[1]), "height": int(originalFormat.shape[0]), "file_name": inputFileName})

        pan_format = np.zeros((originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8)

        segmentIds = np.unique(originalFormat)
        segmInfo = []
        for segmentId in segmentIds:
            if segmentId < 1000:
                semanticId = segmentId
                isCrowd = 1
            else:
                semanticId = segmentId // 1000
                isCrowd = 0
            labelInfo = id2label[semanticId]
            categoryId = labelInfo.trainId if useTrainId else labelInfo.id
            if labelInfo.ignoreInEval:
                continue
            if not labelInfo.hasInstances:
                isCrowd = 0
            mask = originalFormat == segmentId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_format[mask] = color
            area = np.sum(mask) # segment area computation
            # bbox computation for a segment # visualize the boxes # TODO
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]
            if BBOX_VIS:
                bbox_vis.append(bbox)

            segmInfo.append({"id": int(segmentId),              # this is in the format of : id * 1000 + instanceId
                             "category_id": int(categoryId),    # trainid (0,1,2,...,15)
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": isCrowd})

        annotations.append({'image_id': imageId,
                            'file_name': outputFileName,
                            "segments_info": segmInfo})

        Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))
        # print('file saved : {}'.format(os.path.join(panopticFolder, outputFileName)))

        if BBOX_VIS:
            img = Image.fromarray(pan_format)
            ax[0].imshow(img)
            ax[1].imshow(img)
            for bb in bbox_vis:
                x = bb[0]
                y = bb[1]
                w = bb[2]
                h = bb[3]
                rect = patches.Rectangle((x, y), w, h, linewidth=lineWidth, edgecolor=cc, facecolor='none')
                ax[1].add_patch(rect)
            file_name = os.path.join(outputFolder, '{}_bbox.png'.format(imageId))
            print('file_name: {}'.format(file_name))
            plt.savefig(file_name, dpi=300)
            plt.close()

        print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(file_list)), end=' ')
        sys.stdout.flush()

    print("\nSaving the json file {}".format(outFile))
    d = {'images': images, 'annotations': annotations}
    with open(outFile, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-folder",
                        dest="synthiaPath",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)

    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)

    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")

    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train", "test"],
                        type=str)
    args = parser.parse_args()

    # TODO: USER INPUTS BELOW
    crowd_region_threshold = 0
    # crowd_region_threshold = 2000
    # crowd_region_threshold = 1000
    DEBUG = False
    if not DEBUG:
        args.synthiaPath = '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES/GT/panoptic-labels-pklfiles-crowdth-{}-for-daformer/'.format(crowd_region_threshold)
        args.outputFolder = '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES/GT/panoptic-labels-crowdth-{}-for-daformer/'.format(crowd_region_threshold)
    else:
        args.synthiaPath = '/home/suman/Downloads/work_march_23_2022/synthia_gt_visual/pickle_files-crowdth-{}'.format(crowd_region_threshold)
        args.outputFolder = '/home/suman/Downloads/work_march_23_2022/synthia_gt_visual/png_files-crowdth-{}'.format(crowd_region_threshold)
    if not os.path.exists(args.outputFolder):
        os.makedirs(args.outputFolder)

    convert2panoptic(args.synthiaPath, args.outputFolder)


# call the main
if __name__ == "__main__":
    main()
