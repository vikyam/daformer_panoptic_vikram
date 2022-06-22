import argparse
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime

import torch
from experiments import generate_experiment_cfgs
from mmcv import Config, get_git_hash
from tools import train

from mmseg.utils.collect_env import gen_code_archive
from tools import test_panoptic
from mmcv.utils import DictAction



def run_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)

def get_exp_path(work_dir, expid, exp_name):
    exp_folder = f'euler-exp{expid}/{exp_name}'
    exp_path = f'{work_dir}/{exp_folder}'
    return exp_path

def get_work_dir(machine):
    if machine == 'euler':
        work_dir = '/cluster/work/cvl/susaha/experiments/daformer_panoptic_experiments'
    else:
        work_dir = '/media/suman/CVLHDD/apps/experiments/daformer_panoptic_experiments'
    return work_dir

def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('--config', default=CONFIG_FILE, help='test config file path')
    parser.add_argument('--checkpoint', default=CHECKPOINT_FILE, help='checkpoint file')
    parser.add_argument('--panop_eval_folder', default=panop_eval_folder, help='checkpoint file')
    parser.add_argument('--panop_eval_temp_folder_name', default=panop_eval_temp_folder_name, help='checkpoint file')
    parser.add_argument('--panop_eval_temp_folder', default=panop_eval_temp_folder, help='checkpoint file')
    parser.add_argument('--exp_path', default=exp_path, help='exp path')
    parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--format-only', action='store_true', help='Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server')
    parser.add_argument('--eval', default=EVAL,type=str, nargs='+', help='evaluation metrics, which depends on the dataset, e.g., "mIoU" for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', default=SHOW_DIR, help='directory where painted images will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers, available when gpu_collect is not specified')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--opacity', type=float, default=OPACITY, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

if __name__ == '__main__':

    # machine = 'local'
    machine = 'euler'

    # DEBUG = True
    DEBUG = False

    expid = 1004
    # eval_type='daformer'
    eval_type = 'panop_deeplab'
    # gpu_mtotal = 23000
    gpu_mtotal = 11000
    total_eval_time = '4:00'
    mem_per_cpu = 9000
    n_cpus = 16
    #
    JOB_DIR = 'jobs'
    GEN_CONFIG_DIR = 'configs/generated/'
    REMOTE_JOB_DIR = '/cluster/home/susaha/code/CVPR2022/daformer_panoptic/jobs'
    WORKDIR = get_work_dir(machine)
    cfg = {}
    cfg['n_gpus'] = 1
    exp_name = '220425_1752_syn2cs_dacs_panoptic_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_8e844' # TODO
    exp_path = get_exp_path(WORKDIR, expid, exp_name)
    CONFIG_FILE = os.path.join(exp_path, '{}.json'.format(exp_name))
    CHECKPOINT_FILE = os.path.join(exp_path, 'latest.pth')
    SHOW_DIR = ''
    EVAL = 'mIoU'
    OPACITY = 1.0
    panop_eval_folder = os.path.join(exp_path, 'panoptic_eval')
    panop_eval_temp_folder_name = 'panop_eval_25-04-2022_20-45-50-049418' # TODO
    panop_eval_temp_folder = os.path.join(panop_eval_folder, panop_eval_temp_folder_name)

    data_root = 'data/cityscapes/'
    gt_dir = os.path.join(data_root, 'gtFine', 'val')
    if DEBUG:
        ann_dir = 'gtFine_panoptic_debug/cityscapes_panoptic_val'
    else:
        ann_dir = 'gtFine_panoptic/cityscapes_panoptic_val'
    gt_dir_panop = os.path.join(data_root, ann_dir.split('/')[0])

    eval_kwargs = {}
    eval_kwargs['eval_kwargs'] = dict(interval=0,
                                      metric='mIoU',
                                      eval_type=eval_type,
                                      panop_eval_folder=panop_eval_folder,
                                      panop_eval_temp_folder=panop_eval_temp_folder,
                                      dataset_name='Cityscapes',
                                      gt_dir=gt_dir,
                                      debug=DEBUG,
                                      num_samples_debug=12,
                                      gt_dir_panop=gt_dir_panop)

    # child_cfg_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' f'{exp_name}_{str(uuid.uuid4())[:5]}'
    args = parse_args()
    cfg = vars(args)
    cfg['eval_kwargs'] = eval_kwargs
    cfg['exp_path'] = exp_path
    cfg_out_file = os.path.join(GEN_CONFIG_DIR, "panoptic_eval.json")
    # os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
    # assert not os.path.isfile(cfg_out_file)
    with open(cfg_out_file, 'w') as of:
        json.dump(cfg, of, indent=4)

    config_files = []

    if machine == 'local':
        test_panoptic.main([cfg_out_file])
        torch.cuda.empty_cache()

    elif machine ==  'euler':
        snapshot_name = datetime.now().strftime('%y%m%d_%H%M%S')
        code_archive = gen_code_archive(JOB_DIR, f'{snapshot_name}.tar.gz')  # compress the code folder to a tar file and save it under job/ folder
        # Upload to remote
        rsync(code_archive, f'eu:{REMOTE_JOB_DIR}/')  # copy the code tar file to euler
        # Generate submission script
        with open('euler_template_eval.sh', 'r') as f:
            submit_template_str = f.read()
        exec_cmd = 'python tools/test_panoptic.py $CONFIG'

        # here you pass argument to the euler_template.sh
        submit_str = submit_template_str.format(
            job_name=exp_name,                                       # '220419_1311_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_01a50'
            code=code_archive.replace(JOB_DIR, REMOTE_JOB_DIR),         # '/cluster/home/susaha/code/CVPR2022/daformer_panoptic/jobs/220419_131154.tar.gz'
            cfg_file=cfg_out_file,                                   # 'configs/generated//euler-exp1003/220419_1311_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_01a50.json'
            n_gpus=1,                                       # 1
            gpu_mtotal=gpu_mtotal,                               # 23000 or 10240
            total_train_time=total_eval_time,                       # '120:00',
            n_cpus=n_cpus, # 12, # 16,                                                  # daformer uses 6
            mem_per_cpu=mem_per_cpu, # 9000,                                           # daformer uses 6000
            work_dir=WORKDIR,
            exec_cmd=exec_cmd)                                          # 'python tools/train.py $CONFIG'

        submit_file = os.path.join(JOB_DIR, f'{snapshot_name}_eval_submit_{0}.sh') # 'jobs/220419_131154_submit_0.sh'

        with open(submit_file, 'w') as f:
            f.write(submit_str)
        rsync(submit_file, f'eu:{REMOTE_JOB_DIR}/')
        print('Submit job {}'.format(exp_name))
        # sub_out = subprocess.run([f'{submit_file}'], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        sub_out = subprocess.run([f'ssh eu bsub < {submit_file}'], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        log_str = f"\n{exp_name}\n{submit_file}\n{sub_out}\n\n"
        print(log_str)
        with open('euler_eval_log.txt', 'a') as fh:
            fh.write(log_str)


