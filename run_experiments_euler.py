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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
To submit job on euler, you don't need to login to euler
you can use the following command and it will submit the job to euler
the ssh and submission process has been fully automated by this script
$ python run_experiments_euler.py --machine euler --exp <exp id>
OR
you can directly submit it from pycharm

the results are saved at a folder name like: work_dirs/local-exp1003/220420_1404_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_3e1a5
the following output files are saved
    -rw-rw-r-- 1 suman suman  72K Apr 20 14:06 20220420_140447.log
    -rw-rw-r-- 1 suman suman  22K Apr 20 14:06 20220420_140447.log.json
    -rw-rw-r-- 1 suman suman 6.3K Apr 20 14:04 220420_1404_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_3e1a5.json
    -rw-rw-r-- 1 suman suman 258K Apr 20 14:04 code.tar.gz
    -rw-rw-r-- 1 suman suman 505M Apr 20 14:06 iter_30.pth
    lrwxrwxrwx 1 suman suman   11 Apr 20 14:06 latest.pth -> iter_30.pth

'''

def run_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    # group = parser.add_mutually_exclusive_group(required=True) # TODO: original
    group.add_argument(
        '-exp',
        # '--exp', # TODO: original
        type=int,
        default=1004, # TODO:
        help='Experiment id as defined in experiment.py',
    )
    group.add_argument(
        '--config',
        default=None,
        help='Path to config file',
    )
    # parser.add_argument('-machine', type=str, choices=['local', 'euler'], default='local')  # TODO: toggle between debug to actual
    parser.add_argument('--machine', type=str, choices=['local', 'euler'], default='euler') # TODO: toggle between debug to actual

    # parser.add_argument('-debug', action='store_true', default=True) # TODO: toggle between debug to actual
    parser.add_argument('-debug', action='store_true', default=False) # TODO: toggle between debug to actual
    # parser.add_argument('--debug', action='store_true') # TODO: original

    parser.add_argument('--startup-test', action='store_true')

    parser.add_argument('-dry', action='store_true', default=False)
    # parser.add_argument('--dry', action='store_true') # TODO: original

    args = parser.parse_args()
    assert (args.config is None) != (args.exp is None), \
        'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    # the code is first compressed into a tar.gz file and then copied to eu:$REMOTE_JOB_DIR
    REMOTE_JOB_DIR = '/cluster/home/susaha/code/CVPR2022/daformer_panoptic/jobs'
    WORKDIR='/cluster/work/cvl/susaha/experiments/daformer_panoptic_experiments'
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        exp_name = f'{args.machine}-exp{args.exp}'
        if args.startup_test:
            exp_name += '-startup'
        cfgs = generate_experiment_cfgs(args.exp)
        # Generate Configs
        for i, cfg in enumerate(cfgs):
            cfg['debug'] = args.debug
            if args.debug: # Set debug level params in this block below TODO:
                debug_interval = []
                debug_interval.append(10000)        # cfg['evaluation']['interval']
                debug_interval.append(10000)    # cfg['checkpoint_config']['interval']
                debug_interval.append(10000)    # cfg.setdefault('uda', {})['debug_img_interval']
                torch.autograd.set_detect_anomaly(True)
                cfg.setdefault('log_config', {})['interval'] = 1            # display the train log with this interval
                cfg['evaluation']['debug'] = args.debug
                cfg['evaluation']['interval'] = debug_interval[0]                          # evalute the model with this interval
                # this is fixed since the folder /media/suman/DATADISK2/apps/datasets/da_former_datasets/data/cityscapes/gtFine_panoptic_debug has only 12 files
                cfg['evaluation']['num_samples_debug'] = 12                  # use these many examples during evaluation
                cfg['checkpoint_config']['interval'] = debug_interval[1]              # save checkpoint with this interval
                # if 'dacs' in cfg['name'] or 'fda' in cfg['name'] or 'minent' in cfg['name'] or 'advseg' in cfg['name']:
                cfg.setdefault('uda', {})['debug_img_interval'] = debug_interval[2]        # dump the visual results with this interval
                cfg['data']['workers_per_gpu'] = 0
                if args.exp == 1002 or args.exp == 1004:
                    cfg['data']['samples_per_gpu'] = 1 # TODO

            if args.startup_test:
                cfg['log_level'] = logging.ERROR
                cfg['runner'] = dict(type='IterBasedRunner', max_iters=2)
                cfg['evaluation']['interval'] = 100
                cfg['checkpoint_config'] = dict(
                    by_epoch=False, interval=100, save_last=False)
            # Generate Config File
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                          f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local': # Execute on Local machine # TODO
        for i, cfg in enumerate(cfgs):
            if args.startup_test and cfg['seed'] != 0:
                continue
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])
            torch.cuda.empty_cache()
    elif args.machine == 'euler': # Execute on Euler machine # TODO
        # Generate Code Snapshot
        snapshot_name = datetime.now().strftime('%y%m%d_%H%M%S')
        code_archive = gen_code_archive(JOB_DIR, f'{snapshot_name}.tar.gz') # compress the code folder to a tar file and save it under job/ folder
        # Upload to remote
        rsync(code_archive, f'eu:{REMOTE_JOB_DIR}/') # copy the code tar file to euler
        # rsync(code_archive, f'susaha@euler.ethz.ch:{REMOTE_JOB_DIR}/') # TODO: Original
        for i, cfg in enumerate(cfgs):
            # Generate submission script
            with open('euler_template.sh', 'r') as f:
                submit_template_str = f.read()
            if cfg['n_gpus'] > 1:
                exec_cmd = f'bash tools/dist_train.sh $CONFIG {cfg["n_gpus"]}'
            else:
                exec_cmd = 'python tools/train.py $CONFIG'

            # here you pass argument to the euler_template.sh
            submit_str = submit_template_str.format(
                job_name=cfg['name'],                                       # '220419_1311_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_01a50'
                code=code_archive.replace(JOB_DIR, REMOTE_JOB_DIR),         # '/cluster/home/susaha/code/CVPR2022/daformer_panoptic/jobs/220419_131154.tar.gz'
                cfg_file=config_files[i],                                   # 'configs/generated//euler-exp1003/220419_1311_syn2cs_dacs_panoptic_dlv2pan_r101v1c_poly10dapnet_s0_01a50.json'
                n_gpus=cfg['n_gpus'],                                       # 1
                gpu_mtotal=cfg['gpu_mtotal'],                               # 23000 or 10240
                total_train_time='24:00',
                n_cpus=16,                                                  # daformer uses 6
                mem_per_cpu=9000,                                           # daformer uses 6000
                work_dir=WORKDIR,
                exec_cmd=exec_cmd)                                          # 'python tools/train.py $CONFIG'

            submit_file = os.path.join(JOB_DIR, f'{snapshot_name}_submit_{i}.sh') # 'jobs/220419_131154_submit_0.sh'

            with open(submit_file, 'w') as f:
                f.write(submit_str)
            rsync(submit_file, f'eu:{REMOTE_JOB_DIR}/')
            # rsync(submit_file, f'lhoyer@euler.ethz.ch:{REMOTE_JOB_DIR}/')
            # Submit job
            if not args.dry:
                print('Submit job {}'.format(cfg['name']))
                sub_out = subprocess.run([f'ssh eu bsub < {submit_file}'], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
                # sub_out = subprocess.run([f'ssh lhoyer@euler.ethz.ch bsub < {submit_file}'], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
                log_str = f"\n{cfg['name']}\n{submit_file}\n{sub_out}\n\n"
                print(log_str)
                with open('euler_exp_list.txt', 'a') as fh:
                    fh.write(log_str)
    else:
        raise NotImplementedError(args.machine)
