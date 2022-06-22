#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function, absolute_import, division
import os, sys

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *

# from ctrl.dataset.mapillary_panop_jan04_2022_v2 import pad_with_fixed_AS, resize_with_pad # TODO

def instances2dict(imageFileList, verbose=False, dataset_name=None, rgb2id=None, input_image_size=None, mapillary_dataloading_style='OURS'):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        if 'Mapillary' in dataset_name:
            if mapillary_dataloading_style == 'DADA':
                pass
                # imgNp, new_image_shape = resize_with_pad(input_image_size, img, Image.NEAREST, fill_value=0, is_label=True)
                # # imgNp = pad_with_fixed_AS(input_image_size[0] / input_image_size[1], img, fill_value=0, is_label=False)
                # imgNp = rgb2id(imgNp)
            elif mapillary_dataloading_style == 'OURS':
                # compute the downsample ratio dsr
                w1 = input_image_size[0]  # 1024
                h1 = input_image_size[1]  # 768
                w2, h2 = img.size
                dsr = h1 / h2
                size_new = [int(w2 * dsr), int(h2 * dsr)]
                img = img.resize(size_new, Image.NEAREST)
                imgNp = np.asarray(img, np.float32)
                imgNp = rgb2id(imgNp)
            else:
                NotImplementedError('No implementation Error --> cityscapesscripts/evaluation/instances2dict.py')

        elif 'Cityscapes' in dataset_name:
            # Image as numpy array
            imgNp = np.array(img)
        else:
            NotImplementedError('no implementation found at def instances2dict(...) --> cityscapesscripts/evaluation/instances2dict.py')

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            instanceObj = Instance(imgNp, instanceId)

            instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
