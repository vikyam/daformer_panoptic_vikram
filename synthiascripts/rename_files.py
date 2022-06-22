import os.path

'''
cddaformer
source ~/venv/daformer/bin/activate
python synthiascripts/rename_files.py
'''
src_dir = '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES/GT/panoptic-labels-crowdth-0-for-daformer/synthia_panoptic'
trg_dir = '/media/suman/DATADISK2/apps/datasets/da_former_datasets/data/synthia/panoptic-labels-crowdth-0-for-daformer/synthia_panoptic'
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)
flist = os.listdir(src_dir)
for f in flist:
    fid = f.split('.')[0]
    strCmd = 'cp {}/{} {}/{}'.format(src_dir, f, trg_dir, '{}_panoptic.png'.format(fid))
    os.system(strCmd)
    print(fid)