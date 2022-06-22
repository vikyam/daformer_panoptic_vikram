#!/bin/bash
#BSUB  -J {job_name}
#BSUB  -N
#BSUB  -oo {work_dir}_log
#BSUB  -W {total_train_time}
#BSUB  -n {n_cpus}
#BSUB  -R "rusage[mem={mem_per_cpu},ngpus_excl_p={n_gpus}]"
#BSUB  -R "select[gpu_mtotal0>={gpu_mtotal}]"
#GPUS: NVIDIAGeForceRTX2080Ti, NVIDIATITANRTX

# exit when any command fails
set -e

CONFIG={cfg_file}
NGPUS={n_gpus}
CODE={code}
WORKDIR=/cluster/work/cvl/susaha/experiments/daformer_panoptic_experiments/
mkdir -p $WORKDIR
PRETRAINED_MODEL_DIR=$WORKDIR"/pretrained"
mkdir -p $PRETRAINED_MODEL_DIR
cp /cluster/work/cvl/susaha/daformer_pretrained_weights/mit_b5.pth $PRETRAINED_MODEL_DIR

echo "Setup environment..."
set_software_stack.sh new
source /cluster/apps/local/env2lmod.sh
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy hdf5/1.10.1 pigz eth_proxy
source ~/venv/daformer/bin/activate
#source ~/venv/venv-mmseg/bin/activate

echo "Copy code..."
rm -rf $TMPDIR/code/
mkdir -p $TMPDIR/code/
cd $TMPDIR/code/
tar -xzf $CODE
rm -rf data work_dirs pretrained
cd ../..
ln -s $TMPDIR/data/         $TMPDIR/code/data
ln -s $WORKDIR              $TMPDIR/code/work_dirs
ln -s $PRETRAINED_MODEL_DIR $TMPDIR/code/pretrained


echo "Copy datasets..."
#cd $TMPDIR
#if [[ ! -d "data/gta/" ]] && [[ $CONFIG == *"gta"* ]]; then
#  echo "Copy gta..."
#  mkdir -p data/gta/
#  cd data/gta/
#  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/gta_seg/images.tar -C .
#  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/gta_seg/labels.tar -C .
#  cd $TMPDIR/code/
#  python tools/convert_datasets/gta.py data/gta --nproc {n_cpus}
#fi

cd $TMPDIR
if [ ! -d "data/cityscapes/" ]; then
  echo "Copy cityscapes..."
  mkdir -p data/cityscapes/
  cd data/cityscapes/
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/gtFine.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/gtFine_panoptic.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/gtFine_panoptic_debug.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/leftImg8bit.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/sample_class_stats_dict.json.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/sample_class_stats.json.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/samples_with_class.json.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/train.txt.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/cityscapes/val.txt.tar -C .
#  mv gtFine_trainvaltest/gtFine gtFine/
#  cd $TMPDIR/code/
#  python tools/convert_datasets/cityscapes.py data/cityscapes --nproc {n_cpus}
fi

cd $TMPDIR
if [[ ! -d "data/synthia/" ]] && [[ $CONFIG == *"syn"* ]]; then
  echo "Copy synthia..."
  mkdir -p data/synthia/
  cd data/synthia/
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/Depth.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/panoptic-labels-crowdth-0-for-daformer.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/RGB.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/sample_class_stats_dict.json.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/sample_class_stats.json.tar -C .
  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/samples_with_class.json.tar -C .
#  tar -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/RGB.tar -C .
#  tar -I pigz -xf /cluster/work/cvl/susaha/dataset/daformer_datasets/data/synthia/GT.tar.gz -C .
#  cd $TMPDIR/code/
#  python tools/convert_datasets/synthia.py $TMPDIR/data/synthia/ --nproc 8
fi

echo "Run code..."
PYTHONPATH=$PYTHONPATH:$TMPDIR/code/
cd $TMPDIR/code/
echo "{exec_cmd}"
{exec_cmd}
