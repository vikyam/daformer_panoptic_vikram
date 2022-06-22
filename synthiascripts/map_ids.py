# _CITYSCAPES_THING_LIST = [24, 25, 26, 27, 28, 31, 32, 33] # note these are ids and not the trainIds

def get_map_s2c_19cls():
    '''
       retruns a dictionary, where keys are synthia class ids, and values are cityscapes class ids
       to see the synthia class ids, refer to synthiascripts/synthia-rand-cvpr16-readme.txt
       to see the cityscapes class ids, refer to synthiascripts/cityscapes_labels_16cls.py
       [10, 17, 8, 18, 19, 20, 12, 11]
       '''
    map_synthia_to_cityscapes = {
    # synthiaId : cityscapesId
        3: 7,  # road
        4: 8,  # sidewalk
        2: 11,  # building
        21: 12,  # wall
        5: 13,  # fence
        7: 17,  # pole
        15: 19,  # traffic light
        9: 20,  # traffic sign
        6: 21,  # vegetation
        16: 22,  # terrain
        1: 23,  # sky
        10: 24,  # cityscapes:person or synthia:pedestrian  - thing class
        17: 25,  # rider - thing class
        8: 26,  # car - thing class
        18: 27,  # truck - thing class
        19: 28,  # bus - - thing class
        20: 31,  # train - thing class
        12: 32,  # motorcycle - thing class
        11: 33  # bicycle - thing class
    }
    return map_synthia_to_cityscapes


def get_map_s2c():
    '''
       retruns a dictionary, where keys are synthia class ids, and values are cityscapes class ids
       to see the synthia class ids, refer to synthiascripts/synthia-rand-cvpr16-readme.txt
       to see the cityscapes class ids, refer to synthiascripts/cityscapes_labels_16cls.py
       [10, 17, 8, 19, 12, 11]
       '''
    map_synthia_to_cityscapes = {
    # synthiaId : cityscapesId
        3: 7,  # road
        4: 8,  # sidewalk
        2: 11,  # building
        21: 12,  # wall
        5: 13,  # fence
        7: 17,  # pole
        15: 19,  # traffic light
        9: 20,  # traffic sign
        6: 21,  # vegetation
        1: 23,  # sky
        10: 24,  # person / Pedestrian  # THING
        17: 25,  # rider                # THING
        8: 26,  # car                   # THING
        19: 28,  # bus                  # THING
        12: 32,  # motorcycle           # THING
        11: 33,  # bicycle              # THING
    }
    return map_synthia_to_cityscapes


def get_map(num_classes=16):
    if num_classes == 16:
        map_synthiaId_to_trainId = {
                3: 0,  # road
                4: 1,  # Sidewalk
                2: 2,  # Building
                21: 3,  # Wall
                5: 4,  # Fence
                7: 5,  # Pole
                15: 6,  # Traffic light
                9: 7,  # Traffic sign
                6: 8,  # Vegetation
                1: 9,  # sky
                10: 10,  # Pedestrian or person
                17: 11,  # Rider
                8: 12,  # Car
                19: 13,  # Bus
                12: 14,  # Motorcycle
                11: 15,  # Bicycle
            }

    # elif num_classes == 7:
    #     map_synthia_to_cityscapes = {
    #         # CLASS NAME    : GROUP NAME IN 7 CLASS SETTING
    #         1: 4,  # sky           : SKY
    #         2: 1,  # Building      : CONSTRUCTION
    #         3: 0,  # Road          : FLAT
    #         4: 0,  # Sidewalk      : FLAT
    #         5: 1,  # Fence         : CONSTRUCTION
    #         6: 3,  # Vegetation    : NATURE
    #         7: 2,  # Pole          : OBJECT
    #         8: 6,  # Car           : VEHICLE
    #         9: 2,  # Traffic sign  : OBJECT
    #         10: 5,  # Pedestrian    : HUMAN
    #         11: 6,  # Bicycle       : VEHICLE
    #         15: 2,  # Traffic light : OBJECT
    #         22: 0}  # Lanemarking   : FLAT
    else:
        raise NotImplementedError(f"Not yet supported {num_classes} classes")

    return map_synthiaId_to_trainId